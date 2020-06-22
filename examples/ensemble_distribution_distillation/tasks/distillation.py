import os
from types import MethodType

import torch

from examples.ensemble_distribution_distillation.utils import prob_parametrization, freeze_module_params
from fairseq import options, checkpoint_utils
from fairseq.data import data_utils
from fairseq.data.data_utils import collate_tokens
from fairseq.tasks import register_task
from fairseq.tasks.translation import TranslationTask
from fairseq.uncertainty import compute_token_dirichlet_uncertainties, compute_sequence_dirichlet_uncertainties


@register_task('distillation')
class DistillationTask(TranslationTask):
    @staticmethod
    def add_args(parser):
        TranslationTask.add_args(parser)
        parser.add_argument('--ensemble-paths', help='Paths to ensemble models for distillation')
        parser.add_argument('--anneal-start', type=int, help='First update from which to start temperature annealing')
        parser.add_argument('--anneal-end', type=int, help='Last update for annealing')
        parser.add_argument('--init-from-model', type=int, help='Model index in ensemble_paths to use for initialization')
        parser.add_argument('--freeze-weights-until', type=int, help='Freeze encoder/decoder weights until a given step')
        parser.add_argument('--init-temp', type=float, default=10)
        parser.add_argument('--final-temp', type=float, default=1)
        parser.add_argument('--init-xent-weight', type=float, default=0)
        parser.add_argument('--final-xent-weight', type=float, default=0)
        parser.add_argument('--parametrization', choices=prob_parametrization.keys(), default='exp')

    def __init__(self, args, src_dict, tgt_dict, models):
        super().__init__(args, src_dict, tgt_dict)
        self.ensemble = models

        self.anneal_start = args.anneal_start
        self.anneal_end = args.anneal_end

        self.init_temp = args.init_temp
        self.final_temp = args.final_temp
        self.temp = args.init_temp

        self.init_xent_weight = args.init_xent_weight
        self.final_xent_weight = args.final_xent_weight
        self.xent_weight = args.init_xent_weight

        self.freeze_weights_until = args.freeze_weights_until
        self.unfreeze_model = self.freeze_weights_until is not None and self.freeze_weights_until > 0
        self.parametrization = args.parametrization
        self.criterion = args.criterion
        self.compute_uncertainty = getattr(args, 'compute_uncertainty', False)

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        args.left_pad_source = options.eval_bool(args.left_pad_source)
        args.left_pad_target = options.eval_bool(args.left_pad_target)

        paths = args.data.split(os.pathsep)
        assert len(paths) > 0
        # find language pair automatically
        if args.source_lang is None or args.target_lang is None:
            args.source_lang, args.target_lang = data_utils.infer_language_pair(paths[0])
        if args.source_lang is None or args.target_lang is None:
            raise Exception('Could not infer language pair, please provide it explicitly')

        # load dictionaries
        src_dict = cls.load_dictionary(os.path.join(paths[0], 'dict.{}.txt'.format(args.source_lang)))
        tgt_dict = cls.load_dictionary(os.path.join(paths[0], 'dict.{}.txt'.format(args.target_lang)))
        assert src_dict.pad() == tgt_dict.pad()
        assert src_dict.eos() == tgt_dict.eos()
        assert src_dict.unk() == tgt_dict.unk()

        if args.ensemble_paths is not None:
            # Load ensemble
            print('| loading model(s) from {}'.format(args.ensemble_paths))
            models, _model_args = checkpoint_utils.load_model_ensemble(
                args.ensemble_paths.split(','),
                task=TranslationTask.setup_task(args, **kwargs)
            )
            assert args.init_from_model is None or args.init_from_model < len(models)
            use_cuda = torch.cuda.is_available() and not args.cpu
            # Optimize ensemble for generation (includes setting .eval())
            for model in models:
                model.make_generation_fast_(need_attn=False)
                if args.fp16:
                    model.half()
                if use_cuda:
                    model.cuda()
        else:
            models = []

        return cls(args, src_dict, tgt_dict, models)

    def train_step(self, sample, model, criterion, optimizer, ignore_grad=False):
        """
        Do forward and backward, and return the loss as computed by *criterion*
        for the given *model* and *sample*.

        Args:
            sample (dict): the mini-batch. The format is defined by the
                :class:`~fairseq.data.FairseqDataset`.
            model (~fairseq.models.BaseFairseqModel): the model
            criterion (~fairseq.criterions.FairseqCriterion): the criterion
            optimizer (~fairseq.optim.FairseqOptimizer): the optimizer
            ignore_grad (bool): multiply loss by 0 if this is set to True

        Returns:
            tuple:
                - the loss
                - the sample size, which is used as the denominator for the
                  gradient
                - logging outputs to display while training
        """
        model.train()
        sample = self.compute_ensemble_logits(sample)

        if self.unfreeze_model:
            for p in model.parameters():
                p.requires_grad = True
            self.unfreeze_model = False

        loss, sample_size, logging_output = criterion(model, sample)
        if ignore_grad:
            loss *= 0
        optimizer.backward(loss)
        return loss, sample_size, logging_output

    @torch.no_grad()
    def valid_step(self, sample, model, criterion):
        model.eval()

        sample = self.compute_ensemble_logits(sample)

        loss, sample_size, logging_output = criterion(model, sample)
        return loss, sample_size, logging_output

    def build_generator(self, args):
        if getattr(args, 'score_reference', False):
            from fairseq.sequence_scorer import SequenceScorer
            return SequenceScorer(self.target_dictionary)
        else:
            from fairseq.sequence_generator import SequenceGenerator, SequenceGeneratorWithAlignment
            if getattr(args, 'print_alignment', False):
                return SequenceGeneratorWithAlignment(
                    self.target_dictionary,
                    beam_size=getattr(args, 'beam', 5),
                    max_len_a=getattr(args, 'max_len_a', 0),
                    max_len_b=getattr(args, 'max_len_b', 200),
                    min_len=getattr(args, 'min_len', 1),
                    normalize_scores=(not getattr(args, 'unnormalized', False)),
                    len_penalty=getattr(args, 'lenpen', 1),
                    unk_penalty=getattr(args, 'unkpen', 0),
                    sampling=getattr(args, 'sampling', False),
                    sampling_topk=getattr(args, 'sampling_topk', -1),
                    sampling_topp=getattr(args, 'sampling_topp', -1.0),
                    temperature=getattr(args, 'temperature', 1.),
                    diverse_beam_groups=getattr(args, 'diverse_beam_groups', -1),
                    diverse_beam_strength=getattr(args, 'diverse_beam_strength', 0.5),
                    match_source_len=getattr(args, 'match_source_len', False),
                    no_repeat_ngram_size=getattr(args, 'no_repeat_ngram_size', 0),
                )
            else:
                return SequenceGenerator(
                    self.target_dictionary,
                    beam_size=getattr(args, 'beam', 5),
                    max_len_a=getattr(args, 'max_len_a', 0),
                    max_len_b=getattr(args, 'max_len_b', 200),
                    min_len=getattr(args, 'min_len', 1),
                    normalize_scores=(not getattr(args, 'unnormalized', False)),
                    len_penalty=getattr(args, 'lenpen', 1),
                    unk_penalty=getattr(args, 'unkpen', 0),
                    sampling=getattr(args, 'sampling', False),
                    sampling_topk=getattr(args, 'sampling_topk', -1),
                    sampling_topp=getattr(args, 'sampling_topp', -1.0),
                    temperature=getattr(args, 'temperature', 1.),
                    diverse_beam_groups=getattr(args, 'diverse_beam_groups', -1),
                    diverse_beam_strength=getattr(args, 'diverse_beam_strength', 0.5),
                    match_source_len=getattr(args, 'match_source_len', False),
                    no_repeat_ngram_size=getattr(args, 'no_repeat_ngram_size', 0),
                )

    @torch.no_grad()
    def inference_step(self, generator, models, sample, prefix_tokens=None):
        hypos_sample = generator.generate(models, sample, prefix_tokens=prefix_tokens)

        if self.compute_uncertainty:
            # compute uncertainties
            self.add_uncertainties(sample, hypos_sample, models)

        return hypos_sample

    def add_uncertainties(self, sample, hypos, models):
        if len(models) != 1:
            raise NotImplementedError('Uncertainty estimation for ensemble of distilled models is not implemented')
        model = models[0]

        tokens = collate_tokens([out['tokens'] for sent in hypos for out in sent[:self.args.nbest]],
                                eos_idx=self.tgt_dict.eos(), pad_idx=self.tgt_dict.pad())
        prev_output = collate_tokens([out['tokens'] for sent in hypos for out in sent[:self.args.nbest]],
                                     eos_idx=self.tgt_dict.eos(), pad_idx=self.tgt_dict.pad(), move_eos_to_beginning=True)

        src_tokens = sample['net_input']['src_tokens']
        src_lengths = sample['net_input']['src_lengths']
        prev_tokens = sample['net_input']['prev_output_tokens']

        sample['net_input']['src_tokens'] = torch.repeat_interleave(sample['net_input']['src_tokens'], self.args.nbest, dim=0)
        sample['net_input']['src_lengths'] = torch.repeat_interleave(sample['net_input']['src_lengths'], self.args.nbest, dim=0)
        sample['net_input']['prev_output_tokens'] = prev_output

        logits, attn = model(**sample['net_input'])

        sample['net_input']['src_tokens'] = src_tokens
        sample['net_input']['src_lengths'] = src_lengths
        sample['net_input']['prev_output_tokens'] = prev_tokens

        unnormalized_probs = prob_parametrization[self.parametrization](logits)  # dirichlet parameters
        concentrations = unnormalized_probs.sum(dim=-1, keepdim=True)

        normalized_probs = model.get_normalized_probs((logits, attn), log_probs=False)
        normalized_logprobs = normalized_probs.log()

        mask = (tokens != self.tgt_dict.pad()).type(logits.dtype)
        entropy_of_expected, expected_entropy, mutual_information, epkl, mkl = compute_token_dirichlet_uncertainties(unnormalized_probs,
                                                                                                                     concentrations,
                                                                                                                     normalized_probs)
        log_probs, scores, scores_mkl = compute_sequence_dirichlet_uncertainties(unnormalized_probs, concentrations,
                                                                                 normalized_logprobs, tokens, mask)

        for i, sent in enumerate(hypos):
            for j, hypo in enumerate(sent[:self.args.nbest]):
                ind = i * self.args.nbest + j

                zeros_tensor = torch.zeros_like(mkl[ind])

                hypo['token_uncertainties'] = {
                    'entropy_of_expected': entropy_of_expected[ind],
                    'expected_entropy': expected_entropy[ind],
                    'mutual_information': mutual_information[ind],
                    'EPKL': epkl[ind],
                    'MKL': mkl[ind],
                    'ep_entropy_of_expected': zeros_tensor,
                    'ep_mutual_information': zeros_tensor,
                    'ep_EPKL': zeros_tensor,
                    'ep_MKL': zeros_tensor,
                    'token_DU': zeros_tensor,
                    'token_ep_TU': zeros_tensor,
                    'token_pe_TU': zeros_tensor,
                    'token_ep_MKL': zeros_tensor,
                    'token_pe_MKL': zeros_tensor,

                }

                zero_tensor = zeros_tensor.sum()

                hypo['sequence_uncertainties'] = {
                    'log-prob': log_probs[ind],
                    'pe_entropy_of_expected': entropy_of_expected[ind].mean(),
                    'expected_entropy': expected_entropy[ind].mean(),
                    'mutual_information': mutual_information[ind].mean(),
                    'pe_EPKL': epkl[ind].mean(),
                    'pe_MKL': mkl[ind].mean(),
                    'pe_sTU': scores[ind],
                    'pe_sMKL': scores_mkl[ind],
                    'ep_sTU': zero_tensor,
                    'sDU': zero_tensor,
                    'ep_sMKL': zero_tensor,
                    'ep_entropy_of_expected': zero_tensor,
                    'ep_mutual_information': zero_tensor,
                    'ep_EPKL': zero_tensor,
                    'ep_MKL': zero_tensor
                }

    @torch.no_grad()
    def compute_ensemble_logits(self, sample):
        batch_size, num_tokens = sample['target'].size()
        ens_size, vocab_size = len(self.ensemble), len(self.tgt_dict)
        sample['ensemble_logits'] = torch.empty((batch_size, num_tokens, ens_size, vocab_size),
                                                dtype=torch.half if self.args.fp16 else torch.float,
                                                device='cpu' if self.args.cpu else 'cuda')

        for i, model in enumerate(self.ensemble):
            sample['ensemble_logits'][:, :, i] = model(**sample['net_input'])[0]
        return sample

    def update_step(self, num_updates):
        if num_updates < self.anneal_start:
            self.temp = self.init_temp
            self.xent_weight = self.init_xent_weight
        elif num_updates > self.anneal_end:
            self.temp = self.final_temp
            self.xent_weight = self.final_xent_weight
        else:
            progress = (num_updates - self.anneal_start) / (self.anneal_end - self.anneal_start)
            self.temp = self.init_temp + (self.final_temp - self.init_temp) * progress
            self.xent_weight = self.init_xent_weight + (self.final_xent_weight - self.init_xent_weight) * progress
        if self.freeze_weights_until == num_updates:
            self.unfreeze_model = True

    def build_model(self, args):
        """
        Build the :class:`~fairseq.models.BaseFairseqModel` instance for this
        task.

        Args:
            args (argparse.Namespace): parsed command-line arguments

        Returns:
            a :class:`~fairseq.models.BaseFairseqModel` instance
        """
        from fairseq import models
        model = models.build_model(args, self)
        if args.init_from_model is not None:
            if self.ensemble:
                state_dict = self.ensemble[args.init_from_model].state_dict()
                if not model.decoder.share_input_output_embed:
                    state_dict['decoder.embed_out'] = state_dict['decoder.embed_tokens.weight']
                model.load_state_dict(state_dict, strict=False)
            else:
                print('Warning: ensemble_paths was empty and init_from_model is not None. Parameters were not overwritten')

        if args.freeze_weights_until is not None and args.freeze_weights_until > 0:
            freeze_module_params(model.encoder)
            freeze_module_params(model.decoder)
            if model.decoder.share_input_output_embed:
                model.decoder.embed_tokens.weight.requires_grad = True
            else:
                model.decoder.embed_out.requires_grad = True

        if args.parametrization != 'exp':
            # patching get_normalized_probs, as we may use something other than exp for mapping logits to positive numbers
            def patched_get_normalized_probs(self, net_output, log_probs, sample=None):
                """Get normalized probabilities (or log probs) from a net's output."""

                if hasattr(self, 'adaptive_softmax') and self.adaptive_softmax is not None:
                    raise NotImplementedError()

                logits = net_output[0]
                unnormalized_probs = prob_parametrization[args.parametrization](logits)
                probs = unnormalized_probs / unnormalized_probs.sum(dim=-1, keepdim=True)
                if log_probs:
                    return probs.log()
                else:
                    return probs

            model.get_normalized_probs = MethodType(patched_get_normalized_probs, model)

        return model
