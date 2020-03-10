import torch
from torch.nn import functional as F

from fairseq import utils
from . import FairseqCriterion, register_criterion


class DirichletEnDDLoss(object):
    """Standard Negative Log-likelihood of the ensemble predictions"""

    def __init__(self, smoothing=1e-8, teacher_prob_smoothing=1e-3):
        self.smooth_val = smoothing
        self.tp_scaling = 1 - teacher_prob_smoothing

    def __call__(self, *args):
        return self.forward(*args)

    def forward(self, logits, teacher_logits, temp=1.0):
        alphas = torch.exp(logits / temp)
        precision = torch.sum(alphas, dim=1)

        teacher_probs = F.softmax(teacher_logits / temp, dim=2)
        # Smooth for num. stability:
        probs_mean = 1 / (teacher_probs.size()[2])
        # Subtract mean, scale down, add mean back)
        teacher_probs = self.tp_scaling * (teacher_probs - probs_mean) + probs_mean
        assert torch.all(teacher_probs != 0).item()

        log_teacher_probs_geo_mean = torch.mean(torch.log(teacher_probs + self.smooth_val), dim=1)
        assert torch.all(torch.isfinite(log_teacher_probs_geo_mean)).item()

        # Define the cost in two parts (dependent on targets and independent of targets)
        target_independent_term = torch.sum(torch.lgamma(alphas + self.smooth_val), dim=1) \
                                  - torch.lgamma(precision + self.smooth_val)
        assert torch.all(torch.isfinite(target_independent_term)).item()

        target_dependent_term = - torch.sum((alphas - 1.) * log_teacher_probs_geo_mean, dim=1)
        assert torch.all(torch.isfinite(target_dependent_term)).item()

        cost = target_dependent_term + target_independent_term
        assert torch.all(torch.isfinite(cost)).item()

        return torch.mean(cost) * (temp ** 2)


@register_criterion('sequence_distribution_distillation')
class SequenceDistributionDistillationCritertion(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)
        self.smooth_val = 1e-8
        self.tp_scaling = 1 - 1e-3
        self.temp = 1  # TODO anneal

    def forward(self, model, sample, reduce=True):
        # batch x len x mixture_size x n_tokens
        net_output = model(**sample['net_input'])

        # batch x len x ensemble_size x n_tokens
        ensemble_logits = sample['ensemble_logits']

        loss = self.compute_loss(net_output, ensemble_logits, sample, reduce=reduce, temp=self.temp)

        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    def compute_loss(self, net_output, ensemble_logits, sample, reduce, temp=1.0):
        # TODO add support for mixtures
        # TODO somehow we need to anneal temperature

        logits = net_output[0]

        alphas = torch.exp(logits / temp)
        precision = torch.sum(alphas, dim=-1)

        teacher_probs = F.softmax(ensemble_logits / temp, dim=-1)
        # Smooth for num. stability:
        probs_mean = 1 / teacher_probs.size(-1)
        # Subtract mean, scale down, add mean back)
        teacher_probs = self.tp_scaling * (teacher_probs - probs_mean) + probs_mean
        assert torch.all(teacher_probs != 0).item()

        log_teacher_probs_geo_mean = torch.mean(torch.log(teacher_probs + self.smooth_val), dim=-2)
        assert torch.all(torch.isfinite(log_teacher_probs_geo_mean)).item()

        # Define the cost in two parts (dependent on targets and independent of targets)
        target_independent_term = torch.sum(torch.lgamma(alphas + self.smooth_val), dim=-2) \
                                  - torch.lgamma(precision + self.smooth_val)
        assert torch.all(torch.isfinite(target_independent_term)).item()

        target_dependent_term = - torch.sum((alphas - 1.) * log_teacher_probs_geo_mean, dim=-2)
        assert torch.all(torch.isfinite(target_dependent_term)).item()

        cost = target_dependent_term + target_independent_term
        assert torch.all(torch.isfinite(cost)).item()

        cost *= temp ** 2

        # mask loss for padding tokens
        pad_mask = sample['target'].eq(self.padding_idx)
        cost.masked_fill_(pad_mask, 0.)

        if reduce:
            return torch.sum(cost)
        return cost

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        return {
            'loss': sum(log.get('loss', 0) for log in logging_outputs) / sample_size if sample_size > 0 else 0.,
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }
