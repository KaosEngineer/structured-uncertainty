import math

import torch

from examples.ensemble_distribution_distillation.utils import prob_parametrization, get_dirichlet_parameters
from fairseq import utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.criterions.label_smoothed_cross_entropy import label_smoothed_nll_loss


def compute_mean_forward_kl(model, sample, ensemble_stats, net_output, ignore_index, reduce):
    lprobs = model.get_normalized_probs(net_output, log_probs=True)
    target = model.get_targets(sample, net_output)

    teacher_probs = ensemble_stats['logprobs']

    # average loss over all teacher distributions
    lprobs = lprobs.unsqueeze(2).expand_as(teacher_probs)
    loss = torch.nn.functional.kl_div(lprobs, teacher_probs, reduction='none').mean(2).sum(-1)

    # mask loss for padding tokens
    pad_mask = target.eq(ignore_index)
    loss.masked_fill_(pad_mask, 0.)

    if reduce:
        return torch.sum(loss)
    return loss


@torch.no_grad()
def compute_epkl(ensemble_probs, ensemble_mean_probs, ensemble_logprobs):
    mlog_probs = torch.mean(ensemble_logprobs, dim=2)
    eoe_upper_bound = -torch.sum(ensemble_mean_probs * mlog_probs, dim=-1)

    exe = -torch.mean(torch.sum(ensemble_probs * ensemble_logprobs, dim=-1), dim=2)
    epkl = eoe_upper_bound - exe
    return epkl


@torch.no_grad()
def compute_mkl(ensemble_probs, ensemble_mean_probs, ensemble_logprobs):
    mkl = torch.nn.functional.kl_div(ensemble_logprobs, ensemble_mean_probs.unsqueeze(2).expand_as(ensemble_probs),
                                     reduction='none').sum(3, keepdim=True).mean(2)
    return mkl


@torch.no_grad()
def compute_ensemble_stats(sample):
    ensemble_logits = sample['ensemble_logits']
    ensemble_probs = utils.softmax(ensemble_logits, dim=-1)
    ensemble_mean_probs = ensemble_probs.mean(dim=2)
    ensemble_logprobs = utils.log_softmax(ensemble_logits, dim=-1)

    epkl = compute_epkl(ensemble_probs, ensemble_mean_probs, ensemble_logprobs)
    mkl = compute_mkl(ensemble_probs, ensemble_mean_probs, ensemble_logprobs)

    stats = {
        'probs': ensemble_probs,
        'mean_probs': ensemble_mean_probs,
        'logprobs': ensemble_logprobs,
        'epkl': epkl,
        'mkl': mkl,
    }
    return stats


def compute_xent_nll(model, sample, net_output, reduce, label_smoothing, ignore_index):
    lprobs = model.get_normalized_probs(net_output, log_probs=True)
    target = model.get_targets(sample, net_output)
    xent_loss, nll_loss = label_smoothed_nll_loss(
        lprobs.view(-1, lprobs.size(-1)), target.view(-1, 1), label_smoothing, ignore_index=ignore_index,
        reduce=reduce,
    )
    return xent_loss, nll_loss


class _DistillationCriterionBase(FairseqCriterion):
    def __init__(self, args, task):
        super().__init__(args, task)
        self.label_smoothing = args.label_smoothing
        self.task = task
        self.xent_type = args.xent_type

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        parser.add_argument('--xent-type', choices=('xent', 'forward_kl'), default='xent')

    def forward(self, model, sample, reduce=True):
        xent_weight = self.task.xent_weight

        # batch x len x n_tokens
        net_output = model(**sample['net_input'])

        ensemble_stats = compute_ensemble_stats(sample)

        loss, stats = self.compute_loss(model, net_output, ensemble_stats, sample, reduce=reduce)

        if self.xent_type == 'xent':
            xent_loss, nll_loss = compute_xent_nll(model, sample, net_output, reduce, self.label_smoothing, self.padding_idx)
            total_loss = loss + xent_weight * xent_loss

        elif self.xent_type == 'forward_kl':
            with torch.no_grad():
                xent_loss, nll_loss = compute_xent_nll(model, sample, net_output, reduce, self.label_smoothing, self.padding_idx)

            forward_kl = compute_mean_forward_kl(model, sample, net_output, ignore_index=self.padding_idx, reduce=reduce)
            total_loss = loss + xent_weight * forward_kl
        else:
            raise KeyError

        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']

        logging_output = {
            'loss': (utils.item(total_loss.data) if reduce else total_loss.data),
            'nll_loss': utils.item(nll_loss.data) if reduce else nll_loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
            'xent_weight': xent_weight,
            'ensemble_mkl': utils.item(ensemble_stats['mkl'].sum()),
            'ensemble_epkl': utils.item(ensemble_stats['epkl'].sum()),
            **{key: utils.item(value) for key, value in stats.items()}
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
            'xent_weight': sum(log.get('xent_weight', 0) for log in logging_outputs) / sample_size if sample_size > 0 else 0.,
            'ensemble_mkl': sum(log.get('ensemble_mkl', 0) for log in logging_outputs) / sample_size if sample_size > 0 else 0.,
            'ensemble_epkl': sum(log.get('ensemble_epkl', 0) for log in logging_outputs) / sample_size if sample_size > 0 else 0.,
        }


@register_criterion('mean_reverse_kl_distillation')
class MeanReverseKLCritertion(_DistillationCriterionBase):
    def compute_loss(self, model, net_output, ensemble_stats, sample, reduce):
        probs = model.get_normalized_probs(net_output)
        teacher_log_probs = ensemble_stats['logprobs']

        probs = probs.unsqueeze(2).expand_as(teacher_log_probs)
        loss = torch.nn.functional.kl_div(teacher_log_probs, probs, reduction='none').mean(2).sum(-1)

        # mask loss for padding tokens
        pad_mask = model.get_targets(sample, net_output).eq(self.padding_idx)
        loss.masked_fill_(pad_mask, 0.)

        if reduce:
            return torch.sum(loss), dict()
        return loss, dict()


@register_criterion('reverse_kl_mean_distillation')
class ReverseKLMeanCritertion(_DistillationCriterionBase):
    def compute_loss(self, model, net_output, ensemble_stats, sample, reduce):
        probs = model.get_normalized_probs(net_output)
        avg_teacher_log_probs = torch.log(ensemble_stats['mean_probs'])

        loss = torch.nn.functional.kl_div(avg_teacher_log_probs, probs, reduction='none').sum(-1)

        # mask loss for padding tokens
        pad_mask = model.get_targets(sample, net_output).eq(self.padding_idx)
        loss.masked_fill_(pad_mask, 0.)

        if reduce:
            return torch.sum(loss), dict()
        return loss, dict()


@register_criterion('mean_forward_kl_distillation')
class MeanForwardKLCritertion(_DistillationCriterionBase):
    def compute_loss(self, model, net_output, ensemble_stats, sample, reduce):
        log_probs = model.get_normalized_probs(net_output, log_probs=True)
        teacher_probs = ensemble_stats['probs']

        # average loss over all teacher distributions
        log_probs = log_probs.unsqueeze(2).expand_as(teacher_probs)
        loss = torch.nn.functional.kl_div(log_probs, teacher_probs, reduction='none').mean(2).sum(-1)

        # mask loss for padding tokens
        pad_mask = model.get_targets(sample, net_output).eq(self.padding_idx)
        loss.masked_fill_(pad_mask, 0.)

        if reduce:
            return torch.sum(loss), dict()
        return loss, dict()


@register_criterion('forward_kl_mean_distillation')
class ForwardKLMeanCritertion(_DistillationCriterionBase):
    def compute_loss(self, model, net_output, ensemble_stats, sample, reduce):
        log_probs = model.get_normalized_probs(net_output, log_probs=True)
        avg_teacher_probs = ensemble_stats['mean_probs']

        loss = torch.nn.functional.kl_div(log_probs, avg_teacher_probs, reduction='none').sum(-1)

        # mask loss for padding tokens
        pad_mask = model.get_targets(sample, net_output).eq(self.padding_idx)
        loss.masked_fill_(pad_mask, 0.)

        if reduce:
            return torch.sum(loss), dict()
        return loss, dict()


EPS = 1e-8


@register_criterion('sequence_distribution_distillation')
class SequenceDistributionDistillationCritertion(_DistillationCriterionBase):
    def __init__(self, args, task):
        super().__init__(args, task)
        self.parametrization_func = prob_parametrization[task.parametrization]
        self.topk = args.topk_loss
        self.model_offset = args.model_offset

    @staticmethod
    def add_args(parser):
        _DistillationCriterionBase.add_args(parser)
        parser.add_argument('--topk-loss', default=-1, type=int, metavar='D',
                            help='top-k most likely classes will be considered as separate classes, others will be merged')

    def compute_loss(self, model, net_output, ensemble_stats, sample, reduce):
        temp = self.task.temp

        alphas, precision = get_dirichlet_parameters(model, net_output, self.parametrization_func, add_to_alphas=self.model_offset)

        precision_sum = precision.sum()
        precision_min = precision.min()
        precision_max = precision.max()
        stats = {'precision_mean': precision_sum, 'precision_min': precision_min, 'precision_max': precision_max}

        alphas = alphas / temp
        precision = precision / temp

        teacher_probs = ensemble_stats['probs']
        mean_teacher_probs = ensemble_stats['mean_probs']

        if self.topk != -1:
            # sort average probabilities
            sorted_avg_probs, argsort_inds = mean_teacher_probs.sort(dim=-1, descending=True)
            # get indices of k most likely classes
            highest_probs_inds = argsort_inds[..., :self.topk]
            lowest_probs_inds = argsort_inds[..., self.topk:]

            sizes = (-1, -1, teacher_probs.size(2), -1)

            # take probabilities for classes in top-k, sum for other classes
            probs_in_topk = teacher_probs.gather(3, highest_probs_inds.unsqueeze(2).expand(*sizes))
            probs_not_in_topk = teacher_probs.gather(3, lowest_probs_inds.unsqueeze(2).expand(*sizes)).sum(3, keepdim=True)
            teacher_probs = torch.cat((probs_in_topk, probs_not_in_topk), dim=3)

            # take alphas for classes in top-k, sum for other classes
            alphas_topk = alphas.gather(2, highest_probs_inds)
            alphas_not_topk = alphas.gather(2, lowest_probs_inds).sum(2, keepdim=True)
            alphas = torch.cat((alphas_topk, alphas_not_topk), dim=2)

        mean_teacher_probs = teacher_probs.mean(dim=2, keepdim=True)

        teacher_probs = (temp - 1) / (temp + 1) * mean_teacher_probs + 2 / (temp + 1) * teacher_probs
        log_teacher_probs_geo_mean = torch.mean(torch.log(teacher_probs + EPS), dim=-2)

        # Define the cost in two parts (dependent on targets and independent of targets)
        target_independent_term = (torch.sum(torch.lgamma(alphas + EPS), dim=-1) - torch.lgamma(precision + EPS))
        target_dependent_term = - torch.sum((alphas - 1.) * log_teacher_probs_geo_mean, dim=-1)
        cost = (target_dependent_term + target_independent_term) / temp

        # mask loss for padding tokens
        pad_mask = model.get_targets(sample, net_output).eq(self.padding_idx)
        cost.masked_fill_(pad_mask, 0.)

        if reduce:
            return torch.sum(cost), stats
        return cost, stats

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        base_outputs = _DistillationCriterionBase.aggregate_logging_outputs(logging_outputs)
        sample_size = base_outputs['sample_size']
        number_of_outputs = sum(1 if log.get('precision_mean') is not None else 0 for log in logging_outputs)
        return {
            **base_outputs,
            'precision': sum(log.get('precision_mean', 0) for log in logging_outputs) / sample_size if sample_size > 0 else 0.,
            'precision_min': sum(
                log.get('precision_min', 0) for log in logging_outputs) / number_of_outputs if number_of_outputs > 0 else 0.,
            'precision_max': sum(
                log.get('precision_max', 0) for log in logging_outputs) / number_of_outputs if number_of_outputs > 0 else 0.,
        }


@register_criterion('dirichlet_mediator_distillation')
class DirichletMediatorDistillationCriterion(SequenceDistributionDistillationCritertion):
    def __init__(self, args, task):
        super().__init__(args, task)
        self.target_concentration = args.target_concentration
        self.clip_precision = args.clip_precision
        self.target_offset = args.target_offset

    @staticmethod
    def add_args(parser):
        SequenceDistributionDistillationCritertion.add_args(parser)
        parser.add_argument('--target-concentration', choices=('mkl', 'epkl'), required=True)
        parser.add_argument('--clip-precision', action='store_true')
        parser.add_argument('--target-offset', default=0, type=float)

    def compute_loss(self, model, net_output, ensemble_stats, sample, reduce):
        temp = self.task.temp

        alphas, precision = get_dirichlet_parameters(model, net_output, self.parametrization_func, self.model_offset)

        num_classes = alphas.size(-1)

        if self.target_concentration == 'mkl':
            mkl = ensemble_stats['mkl']
            ensemble_precision = (num_classes - 1) / (2 * mkl + EPS)
        elif self.target_concentration == 'epkl':
            epkl = ensemble_stats['epkl'].unsqueeze(2)
            ensemble_precision = (num_classes - 1) / (epkl + EPS)
        else:
            raise ValueError

        if self.clip_precision:
            torch.clamp(ensemble_precision, min=num_classes, out=ensemble_precision)
            precision = torch.clamp(precision, min=num_classes)
            alphas = alphas / alphas.sum(dim=-1, keepdim=True) * precision

        ensemble_params = ensemble_stats['mean_probs'] * ensemble_precision + self.target_offset
        ensemble_precision += self.target_offset * num_classes

        precision_sum = precision.sum()
        precision_min = precision.min()
        precision_max = precision.max()
        ensemble_precision_sum = ensemble_precision.sum()
        ensemble_precision_min = ensemble_precision.min()
        ensemble_precision_max = ensemble_precision.max()

        stats = {'precision': precision_sum, 'precision_min': precision_min, 'precision_max': precision_max,
                 'ensemble_precision': ensemble_precision_sum, 'ensemble_precision_min': ensemble_precision_min,
                 'ensemble_precision_max': ensemble_precision_max}

        alphas = alphas / temp
        precision = precision / temp
        ensemble_precision /= temp
        ensemble_params /= temp

        if self.topk != -1:
            # sort average probabilities
            sorted_avg_probs, argsort_inds = ensemble_stats['mean_probs'].sort(dim=-1, descending=True)
            # get indices of k most likely classes
            highest_probs_inds = argsort_inds[..., :self.topk]
            lowest_probs_inds = argsort_inds[..., self.topk:]

            # take parameters for classes in top-k, sum for other classes
            params_in_topk = ensemble_params.gather(2, highest_probs_inds)
            params_not_in_topk = ensemble_params.gather(2, lowest_probs_inds).sum(2, keepdim=True)
            ensemble_params = torch.cat((params_in_topk, params_not_in_topk), dim=2)

            # take alphas for classes in top-k, sum for other classes
            alphas_topk = alphas.gather(2, highest_probs_inds)
            alphas_not_topk = alphas.gather(2, lowest_probs_inds).sum(2, keepdim=True)
            alphas = torch.cat((alphas_topk, alphas_not_topk), dim=2)

        target_independent_term = (
                torch.lgamma(ensemble_precision.squeeze(2)) - torch.sum(torch.lgamma(ensemble_params), dim=-1) +
                torch.sum(torch.lgamma(alphas), dim=-1) - torch.lgamma(precision)
        )

        target_dependent_term = torch.sum(
            (ensemble_params - alphas) *
            (torch.digamma(ensemble_params) - torch.digamma(ensemble_precision)),
            dim=-1)

        cost = target_dependent_term + target_independent_term
        # mask loss for padding tokens
        pad_mask = model.get_targets(sample, net_output).eq(self.padding_idx)
        cost.masked_fill_(pad_mask, 0.)

        if reduce:
            return torch.sum(cost), stats
        return cost, stats

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        base_outputs = SequenceDistributionDistillationCritertion.aggregate_logging_outputs(logging_outputs)
        sample_size = base_outputs['sample_size']
        number_of_outputs = sum(1 if log.get('ensemble_precision') is not None else 0 for log in logging_outputs)
        return {
            **base_outputs,
            'ensemble_precision': sum(log.get('ensemble_precision', 0) for log in logging_outputs) / sample_size if sample_size > 0 else 0.,
            'ensemble_precision_min': sum(
                log.get('ensemble_precision_min', 0) for log in logging_outputs) / number_of_outputs if number_of_outputs > 0 else 0.,
            'ensemble_precision_max': sum(
                log.get('ensemble_precision_max', 0) for log in logging_outputs) / number_of_outputs if number_of_outputs > 0 else 0.,
        }
