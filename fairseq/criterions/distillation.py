import torch
from torch.nn import functional as F
import math

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
        self.tp_scaling = 1 - 1e-8
        self.temp = 1

    def forward(self, model, sample, reduce=True):
        # batch x len x n_tokens
        net_output = model(**sample['net_input'])

        # batch x len x ensemble_size x n_tokens
        ensemble_logits = sample['ensemble_logits']

        loss = self.compute_loss(model, net_output, ensemble_logits, sample, reduce=reduce, temp=self.temp)
        nll_loss = self.compute_nll(model, net_output, sample, reduce=reduce)

        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'nll_loss': utils.item(nll_loss.data) if reduce else nll_loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, ensemble_logits, sample, reduce, temp):
        # TODO add support for mixtures

        logits = net_output[0].float()
        ensemble_logits = ensemble_logits.float()

        num_classes = ensemble_logits.size(-1)

        # TODO transform logits with softlimit(x)=5tanh(x), x>0, x, x<0
        # (or C+x/100, x>C)

        alphas = torch.exp(logits / temp)
        precision = torch.sum(alphas, dim=-1)

        probs_mean = 1 / ensemble_logits.size(-1)
        teacher_probs = self.tp_scaling * utils.softmax(ensemble_logits / temp, dim=-1) + (1 - self.tp_scaling) * probs_mean
        # Smooth for num. stability:
        # Subtract mean, scale down, add mean back
        # teacher_probs = self.tp_scaling * (teacher_probs - probs_mean) + probs_mean
        # (or interpolate between true and uniform distributions)
        # teacher_probs = self.tp_scaling * teacher_probs + (1 - self.tp_scaling) * probs_mean
        assert torch.all(teacher_probs != 0).item()
        log_teacher_probs_geo_mean = torch.mean(torch.log(teacher_probs + self.smooth_val), dim=-2)
        assert torch.all(torch.isfinite(log_teacher_probs_geo_mean)).item()

        # Define the cost in two parts (dependent on targets and independent of targets)
        target_independent_term = (torch.sum(torch.lgamma(alphas + self.smooth_val), dim=-1)
                                   - torch.lgamma(precision + self.smooth_val))
        assert torch.all(torch.isfinite(target_independent_term)).item()

        target_dependent_term = - torch.sum((alphas - 1.) * log_teacher_probs_geo_mean, dim=-1)
        assert torch.all(torch.isfinite(target_dependent_term)).item()

        cost = target_dependent_term + target_independent_term
        assert torch.all(torch.isfinite(cost)).item()

        # mask loss for padding tokens
        pad_mask = model.get_targets(sample, net_output).eq(self.padding_idx)
        cost.masked_fill_(pad_mask, 0.)

        if reduce:
            return torch.sum(cost) / num_classes
        return cost

    @torch.no_grad()
    def compute_nll(self, model, net_output, sample, reduce):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1, 1)
        if target.dim() == lprobs.dim() - 1:
            target = target.unsqueeze(-1)
        nll_loss = -lprobs.gather(dim=-1, index=target)
        pad_mask = target.eq(self.padding_idx)
        if pad_mask.any():
            nll_loss.masked_fill_(pad_mask, 0.)
        else:
            nll_loss = nll_loss.squeeze(-1)
        if reduce:
            nll_loss = nll_loss.sum()
        return nll_loss

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
