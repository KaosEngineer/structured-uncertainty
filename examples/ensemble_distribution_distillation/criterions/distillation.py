import math

import torch

from examples.ensemble_distribution_distillation.utils import prob_parametrization, get_dirichlet_parameters
from fairseq import utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.criterions.label_smoothed_cross_entropy import label_smoothed_nll_loss


def compute_mean_forward_kl(lprobs, target, ensemble_logits, ignore_index, reduce):
    teacher_probs = utils.softmax(ensemble_logits, dim=-1)

    # average loss over all teacher distributions
    lprobs = lprobs.unsqueeze(2).expand_as(teacher_probs)
    loss = torch.nn.functional.kl_div(lprobs, teacher_probs, reduction='none').mean(2).sum(-1)

    # mask loss for padding tokens
    pad_mask = target.eq(ignore_index)
    loss.masked_fill_(pad_mask, 0.)

    if reduce:
        return torch.sum(loss)
    return loss


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

        # batch x len x ensemble_size x n_tokens
        ensemble_logits = sample['ensemble_logits']

        loss, stats = self.compute_loss(model, net_output, ensemble_logits, sample, reduce=reduce)

        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs
        target = model.get_targets(sample, net_output)

        if self.xent_type == 'xent':
            xent_loss, nll_loss = label_smoothed_nll_loss(
                lprobs.view(-1, lprobs.size(-1)), target.view(-1, 1), self.label_smoothing, ignore_index=self.padding_idx, reduce=reduce,
            )
            total_loss = loss + xent_weight * xent_loss

        elif self.xent_type == 'forward_kl':
            with torch.no_grad():
                xent_loss, nll_loss = label_smoothed_nll_loss(
                    lprobs.view(-1, lprobs.size(-1)), target.view(-1, 1), self.label_smoothing, ignore_index=self.padding_idx,
                    reduce=reduce,
                )

            forward_kl = compute_mean_forward_kl(lprobs, target, ensemble_logits, ignore_index=self.padding_idx, reduce=reduce)
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
            **stats
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
            'xent_weight': sum(log.get('xent_weight', 0) for log in logging_outputs) / sample_size if sample_size > 0 else 0.
        }


@register_criterion('mean_reverse_kl_distillation')
class MeanReverseKLCritertion(_DistillationCriterionBase):
    def compute_loss(self, model, net_output, ensemble_logits, sample, reduce):
        probs = model.get_normalized_probs(net_output)
        teacher_log_probs = utils.log_softmax(ensemble_logits, dim=-1)

        probs = probs.unsqueeze(2).expand_as(teacher_log_probs)
        loss = torch.nn.functional.kl_div(teacher_log_probs, probs, reduction='none').mean(2).sum(-1)

        # mask loss for padding tokens
        pad_mask = model.get_targets(sample, net_output).eq(self.padding_idx)
        loss.masked_fill_(pad_mask, 0.)

        if reduce:
            return torch.sum(loss)
        return loss, dict()


@register_criterion('reverse_kl_mean_distillation')
class ReverseKLMeanCritertion(_DistillationCriterionBase):
    def compute_loss(self, model, net_output, ensemble_logits, sample, reduce):
        probs = model.get_normalized_probs(net_output)
        avg_teacher_log_probs = torch.log(utils.softmax(ensemble_logits, dim=-1).mean(2))

        loss = torch.nn.functional.kl_div(avg_teacher_log_probs, probs, reduction='none').sum(-1)

        # mask loss for padding tokens
        pad_mask = model.get_targets(sample, net_output).eq(self.padding_idx)
        loss.masked_fill_(pad_mask, 0.)

        if reduce:
            return torch.sum(loss)
        return loss, dict()


@register_criterion('mean_forward_kl_distillation')
class MeanForwardKLCritertion(_DistillationCriterionBase):
    def compute_loss(self, model, net_output, ensemble_logits, sample, reduce):
        log_probs = model.get_normalized_probs(net_output, log_probs=True)
        teacher_probs = utils.softmax(ensemble_logits, dim=-1)

        # average loss over all teacher distributions
        log_probs = log_probs.unsqueeze(2).expand_as(teacher_probs)
        loss = torch.nn.functional.kl_div(log_probs, teacher_probs, reduction='none').mean(2).sum(-1)

        # mask loss for padding tokens
        pad_mask = model.get_targets(sample, net_output).eq(self.padding_idx)
        loss.masked_fill_(pad_mask, 0.)

        if reduce:
            return torch.sum(loss)
        return loss, dict()


@register_criterion('forward_kl_mean_distillation')
class ForwardKLMeanCritertion(_DistillationCriterionBase):
    def compute_loss(self, model, net_output, ensemble_logits, sample, reduce):
        log_probs = model.get_normalized_probs(net_output, log_probs=True)
        avg_teacher_probs = utils.softmax(ensemble_logits, dim=-1).mean(2)

        loss = torch.nn.functional.kl_div(log_probs, avg_teacher_probs, reduction='none').sum(-1)

        # mask loss for padding tokens
        pad_mask = model.get_targets(sample, net_output).eq(self.padding_idx)
        loss.masked_fill_(pad_mask, 0.)

        if reduce:
            return torch.sum(loss)
        return loss, dict


EPS = 1e-8


@register_criterion('sequence_distribution_distillation')
class SequenceDistributionDistillationCritertion(_DistillationCriterionBase):
    def __init__(self, args, task):
        super().__init__(args, task)
        self.parametrization_func = prob_parametrization[task.parametrization]
        self.topk = args.topk_loss

    @staticmethod
    def add_args(parser):
        _DistillationCriterionBase.add_args(parser)
        parser.add_argument('--topk-loss', default=-1, type=int, metavar='D',
                            help='top-k most likely classes will be considered as separate classes, others will be merged')

    def compute_loss(self, model, net_output, ensemble_logits, sample, reduce):
        temp = self.task.temp

        alphas, precision = get_dirichlet_parameters(model, net_output, self.parametrization_func)

        alphas = alphas * temp
        precision = precision * temp

        teacher_probs = utils.softmax(ensemble_logits, dim=-1).float()

        if self.topk != -1:
            # sort average probabilities
            sorted_avg_probs, argsort_inds = teacher_probs.mean(2).sort(dim=-1, descending=True)
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
            return torch.sum(cost)
        return cost


@register_criterion('dirichlet_mediator_distillation')
class DirichletMediatorDistillationCriterion(SequenceDistributionDistillationCritertion):
    def __init__(self, args, task):
        super().__init__(args, task)
        self.target_concentration = args.target_concentration

    @staticmethod
    def add_args(parser):
        SequenceDistributionDistillationCritertion.add_args(parser)
        parser.add_argument('--target-concentration', choices=('mkl', 'epkl'), required=True)

    def compute_loss(self, model, net_output, ensemble_logits, sample, reduce):
        temp = self.task.temp

        alphas, precision = get_dirichlet_parameters(model, net_output, self.parametrization_func)

        num_classes = ensemble_logits.size(3)

        ensemble_probs = utils.softmax(ensemble_logits, dim=-1)
        ensemble_mean_probs = ensemble_probs.mean(dim=2)
        ensemble_logprobs = torch.log(ensemble_probs)

        if self.target_concentration == 'mkl':
            mkl = torch.nn.functional.kl_div(ensemble_logprobs, ensemble_mean_probs.unsqueeze(2).expand_as(ensemble_probs),
                                             reduction='none').sum(3, keepdim=True).mean(2)
            ensemble_precision = (num_classes - 1) / (2 * mkl)
        elif self.target_concentration == 'epkl':
            epkl = 1
            ensemble_precision = (num_classes - 1) / epkl
        else:
            raise ValueError

        precision_sum = (precision * temp).sum().item()
        ensemble_precision_sum = ensemble_precision.sum().item()
        stats = {'precision': precision_sum, 'ensemble_precision': ensemble_precision_sum}

        alphas = alphas / temp
        precision = precision / temp
        ensemble_precision = ensemble_precision / temp

        ensemble_params = ensemble_mean_probs * ensemble_precision

        if self.topk != -1:
            # sort average probabilities
            sorted_avg_probs, argsort_inds = ensemble_mean_probs.sort(dim=-1, descending=True)
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
        base_outputs = _DistillationCriterionBase.aggregate_logging_outputs(logging_outputs)
        sample_size = base_outputs['sample_size']

        precision = sum(log.get('precision', 0) for log in logging_outputs) / sample_size if sample_size > 0 else 0.
        ensemble_precision = sum(log.get('ensemble_precision', 0) for log in logging_outputs) / sample_size if sample_size > 0 else 0.

        return {
            **base_outputs,
            'precision': precision,
            'ensemble_precision': ensemble_precision
        }
