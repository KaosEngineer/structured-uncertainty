import os

import torch

from fairseq import options, checkpoint_utils
from fairseq.data import (
    data_utils,
)
from fairseq.tasks import register_task
from fairseq.tasks.translation import TranslationTask


@register_task('distillation')
class DistillationTask(TranslationTask):
    @staticmethod
    def add_args(parser):
        TranslationTask.add_args(parser)
        parser.add_argument('--ensemble-paths', help='Paths to ensemble models for distillation')
        parser.add_argument('--anneal-start', help='First update from which to start temperature annealing')
        parser.add_argument('--anneal-end', help='Last update for annealing')
        parser.add_argument('--init-temp', default=10)
        parser.add_argument('--final-temp', default=1)

    def __init__(self, args, src_dict, tgt_dict, models):
        super().__init__(args, src_dict, tgt_dict)
        self.ensemble = models
        self.anneal_start = args.anneal_start
        self.anneal_end = args.anneal_end
        self.init_temp = args.init_temp
        self.final_temp = args.final_temp
        self.temp = args.init_temp

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
        print('| [{}] dictionary: {} types'.format(args.source_lang, len(src_dict)))
        print('| [{}] dictionary: {} types'.format(args.target_lang, len(tgt_dict)))

        # Load ensemble
        print('| loading model(s) from {}'.format(args.ensemble_paths))
        models, _model_args = checkpoint_utils.load_model_ensemble(
            args.ensemble_paths.split(','),
            task=TranslationTask.setup_task(args, **kwargs)
        )
        use_cuda = torch.cuda.is_available() and not args.cpu
        # Optimize ensemble for generation (includes setting .eval())
        for model in models:
            model.make_generation_fast_(need_attn=False)
            if args.fp16:
                model.half()
            if use_cuda:
                model.cuda()

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

        criterion.temp = self.temp
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

    def inference_step(self, generator, models, sample, prefix_tokens=None):
        # TODO what if model was trained with mixture of Dir()?
        with torch.no_grad():
            return generator.generate(models, sample, prefix_tokens=prefix_tokens)

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
        elif num_updates > self.anneal_end:
            self.temp = self.final_temp
        else:
            progress = (num_updates - self.anneal_start) / (self.anneal_end - self.anneal_start)
            self.temp = self.init_temp + (self.final_temp - self.init_temp) * progress
