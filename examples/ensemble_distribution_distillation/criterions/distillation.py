import math

import torch

from fairseq import utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.criterions.label_smoothed_cross_entropy import label_smoothed_nll_loss


class _DistillationCriterionBase(FairseqCriterion):
    def __init__(self, args, task):
        super().__init__(args, task)
        self.xent_weight = args.xent_weight
        self.eps = args.label_smoothing

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        parser.add_argument('--xent-weight', default=0, type=float)

    def forward(self, model, sample, reduce=True):
        # batch x len x n_tokens
        net_output = model(**sample['net_input'])

        # batch x len x ensemble_size x n_tokens
        ensemble_logits = sample['ensemble_logits']

        loss = self.compute_loss(model, net_output, ensemble_logits, sample, reduce=reduce)

        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1, 1)

        xent_loss, nll_loss = label_smoothed_nll_loss(
            lprobs, target, self.eps, ignore_index=self.padding_idx, reduce=reduce,
        )

        total_loss = loss + self.xent_weight * xent_loss

        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': (utils.item(total_loss.data) if reduce else total_loss.data),
            'nll_loss': utils.item(nll_loss.data) if reduce else nll_loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        return total_loss, sample_size, logging_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        return {
            'loss': sum(log.get('loss', 0) for log in logging_outputs) / sample_size if sample_size > 0 else 0.,
            'nll_loss': sum(log.get('nll_loss', 0) for log in logging_outputs) / ntokens / math.log(2) if ntokens > 0 else 0.,
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }


@register_criterion('mean_reverse_kl_distillation')
class MeanReverseKLCritertion(_DistillationCriterionBase):
    def compute_loss(self, model, net_output, ensemble_logits, sample, reduce):
        logits = net_output[0]

        probs = utils.softmax(logits, dim=-1)
        teacher_log_probs = utils.log_softmax(ensemble_logits, dim=-1)

        probs = probs.unsqueeze(2).expand_as(teacher_log_probs)
        loss = torch.nn.functional.kl_div(teacher_log_probs, probs, reduction='none').mean(2).sum(-1)

        # mask loss for padding tokens
        pad_mask = model.get_targets(sample, net_output).eq(self.padding_idx)
        loss.masked_fill_(pad_mask, 0.)

        if reduce:
            return torch.sum(loss)
        return loss


@register_criterion('reverse_kl_mean_distillation')
class ReverseKLMeanCritertion(_DistillationCriterionBase):
    def compute_loss(self, model, net_output, ensemble_logits, sample, reduce):
        logits = net_output[0]

        probs = utils.softmax(logits, dim=-1)
        avg_teacher_log_probs = torch.log(utils.softmax(ensemble_logits, dim=-1).mean(2))

        loss = torch.nn.functional.kl_div(avg_teacher_log_probs, probs, reduction='none').sum(-1)

        # mask loss for padding tokens
        pad_mask = model.get_targets(sample, net_output).eq(self.padding_idx)
        loss.masked_fill_(pad_mask, 0.)

        if reduce:
            return torch.sum(loss)
        return loss


@register_criterion('mean_forward_kl_distillation')
class MeanForwardKLCritertion(_DistillationCriterionBase):
    def compute_loss(self, model, net_output, ensemble_logits, sample, reduce):
        logits = net_output[0]

        log_probs = utils.log_softmax(logits, dim=-1)
        teacher_probs = utils.softmax(ensemble_logits, dim=-1)

        # average loss over all teacher distributions
        log_probs = log_probs.unsqueeze(2).expand_as(teacher_probs)
        loss = torch.nn.functional.kl_div(log_probs, teacher_probs, reduction='none').mean(2).sum(-1)

        # mask loss for padding tokens
        pad_mask = model.get_targets(sample, net_output).eq(self.padding_idx)
        loss.masked_fill_(pad_mask, 0.)

        if reduce:
            return torch.sum(loss)
        return loss


@register_criterion('forward_kl_mean_distillation')
class ForwardKLMeanCritertion(_DistillationCriterionBase):
    def compute_loss(self, model, net_output, ensemble_logits, sample, reduce):
        logits = net_output[0]

        log_probs = utils.log_softmax(logits, dim=-1)
        avg_teacher_probs = utils.softmax(ensemble_logits, dim=-1).mean(2)

        loss = torch.nn.functional.kl_div(log_probs, avg_teacher_probs, reduction='none').sum(-1)

        # mask loss for padding tokens
        pad_mask = model.get_targets(sample, net_output).eq(self.padding_idx)
        loss.masked_fill_(pad_mask, 0.)

        if reduce:
            return torch.sum(loss)
        return loss


parametrization_options = {'exp': torch.exp, 'softplus': torch.nn.functional.softplus}


@register_criterion('sequence_distribution_distillation')
class SequenceDistributionDistillationCritertion(_DistillationCriterionBase):

    @staticmethod
    def add_args(parser):
        super().add_args(parser)
        parser.add_argument('--parametrization', choices=parametrization_options.keys(), default='exp')

    def __init__(self, args, task):
        super().__init__(args, task)
        self.smooth_val = 1e-8
        self.temp = 1
        self.parametrization = parametrization_options[args.parametrization]

    def forward(self, model, sample, reduce=True):
        # batch x len x n_tokens
        net_output = model(**sample['net_input'])

        # batch x len x ensemble_size x n_tokens
        ensemble_logits = sample['ensemble_logits']

        loss = self.compute_loss(model, net_output, ensemble_logits, sample, reduce=reduce, temp=self.temp)

        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1, 1)

        xent_loss, nll_loss = label_smoothed_nll_loss(
            lprobs, target, self.eps, ignore_index=self.padding_idx, reduce=reduce,
        )

        total_loss = loss + self.xent_weight * xent_loss

        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': (utils.item(total_loss.data) if reduce else total_loss.data),
            'nll_loss': utils.item(nll_loss.data) if reduce else nll_loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        return total_loss, sample_size, logging_output

    def compute_loss(self, model, net_output, ensemble_logits, sample, reduce, temp):
        # TODO add support for mixtures

        logits = net_output[0].float()
        ensemble_logits = ensemble_logits.float()

        alphas = temp * self.parametrization(logits)
        precision = torch.sum(alphas, dim=-1)

        teacher_probs = utils.softmax(ensemble_logits, dim=-1)
        mean_teacher_probs = teacher_probs.mean(dim=2, keepdim=True)

        teacher_probs = (temp - 1) / (temp + 1) * mean_teacher_probs + 2 / (temp + 1) * teacher_probs
        log_teacher_probs_geo_mean = torch.mean(torch.log(teacher_probs + self.smooth_val), dim=-2)

        # Define the cost in two parts (dependent on targets and independent of targets)
        target_independent_term = (torch.sum(torch.lgamma(alphas + self.smooth_val), dim=-1) - torch.lgamma(precision + self.smooth_val))

        target_dependent_term = - torch.sum((alphas - 1.) * log_teacher_probs_geo_mean, dim=-1)

        cost = (target_dependent_term + target_independent_term) / temp

        # mask loss for padding tokens
        pad_mask = model.get_targets(sample, net_output).eq(self.padding_idx)
        cost.masked_fill_(pad_mask, 0.)

        if reduce:
            return torch.sum(cost)
        return cost
