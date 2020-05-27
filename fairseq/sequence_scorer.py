# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import sys

from fairseq import utils


class SequenceScorer(object):
    """Scores the target for a given source sentence."""

    def __init__(self, tgt_dict, softmax_batch=None):
        self.pad = tgt_dict.pad()
        self.eos = tgt_dict.eos()
        self.softmax_batch = softmax_batch or sys.maxsize
        assert self.softmax_batch > 0

    @torch.no_grad()
    def generate(self, models, sample, **kwargs):
        """Score a batch of translations."""
        net_input = sample['net_input']

        def batch_for_softmax(dec_out, target):
            # assumes decoder_out[0] is the only thing needed (may not be correct for future models!)
            first, rest = dec_out[0], dec_out[1:]
            bsz, tsz, dim = first.shape
            if bsz * tsz < self.softmax_batch:
                yield dec_out, target, True
            else:
                flat = first.contiguous().view(1, -1, dim)
                flat_tgt = target.contiguous().view(flat.shape[:-1])
                s = 0
                while s < flat.size(1):
                    e = s + self.softmax_batch
                    yield (flat[:, s:e],) + rest, flat_tgt[:, s:e], False
                    s = e

        def gather_target_probs(probs, target):
            probs = probs.gather(
                dim=2,
                index=target.unsqueeze(-1),
            )
            return probs

        orig_target = sample['target']

        # compute scores for each model in the ensemble
        avg_probs = None
        avg_attn = None
        for model in models:
            model.eval()
            decoder_out = model(**net_input)
            attn = decoder_out[1]
            if type(attn) is dict:
                attn = attn.get('attn', None)

            batched = batch_for_softmax(decoder_out, orig_target)
            probs, idx = None, 0
            for bd, tgt, is_single in batched:
                sample['target'] = tgt
                curr_prob = model.get_normalized_probs(bd, log_probs=len(models) == 1, sample=sample).data
                if is_single:
                    probs = gather_target_probs(curr_prob, orig_target)
                else:
                    if probs is None:
                        probs = curr_prob.new(orig_target.numel())
                    step = curr_prob.size(0) * curr_prob.size(1)
                    end = step + idx
                    tgt_probs = gather_target_probs(curr_prob.view(tgt.shape + (curr_prob.size(-1),)), tgt)
                    probs[idx:end] = tgt_probs.view(-1)
                    idx = end
                sample['target'] = orig_target

            probs = probs.view(sample['target'].shape)

            if avg_probs is None:
                avg_probs = probs
            else:
                avg_probs.add_(probs)
            if attn is not None and torch.is_tensor(attn):
                attn = attn.data
                if avg_attn is None:
                    avg_attn = attn
                else:
                    avg_attn.add_(attn)
        if len(models) > 1:
            avg_probs.div_(len(models))
            avg_probs.log_()
            if avg_attn is not None:
                avg_attn.div_(len(models))

        bsz = avg_probs.size(0)
        hypos = []
        start_idxs = sample['start_indices'] if 'start_indices' in sample else [0] * bsz
        for i in range(bsz):
            # remove padding from ref
            ref = utils.strip_pad(sample['target'][i, start_idxs[i]:], self.pad) \
                if sample['target'] is not None else None
            tgt_len = ref.numel()
            avg_probs_i = avg_probs[i][start_idxs[i]:start_idxs[i] + tgt_len]
            score_i = avg_probs_i.sum() / tgt_len
            if avg_attn is not None:
                avg_attn_i = avg_attn[i]
                alignment = utils.extract_hard_alignment(
                    avg_attn_i,
                    sample['net_input']['src_tokens'][i],
                    sample['target'][i],
                    self.pad,
                    self.eos,
                )
            else:
                avg_attn_i = alignment = None
            hypos.append([{
                'tokens': ref,
                'score': score_i,
                'attention': avg_attn_i,
                'alignment': alignment,
                'positional_scores': avg_probs_i,
            }])
        return hypos


class SequenceScorerWithUncertainty(SequenceScorer):
    """Scores the target for a given source sentence. Additionally yields measures of uncertainty"""

    @torch.no_grad()
    def generate(self, models, sample, **kwargs):
        # Uncertainties are derived using an ensemble of models, so we have to have more than 1 model...
        assert len(models) > 1

        from fairseq.uncertainty import token_uncertainties, aep_uncertainty
        """Score a batch of translations."""
        net_input = sample['net_input']

        def batch_for_softmax(dec_out, target):
            # assumes decoder_out[0] is the only thing needed (may not be correct for future models!)
            first, rest = dec_out[0], dec_out[1:]
            bsz, tsz, dim = first.shape
            if bsz * tsz < self.softmax_batch:
                yield dec_out, target, True
            else:
                flat = first.contiguous().view(1, -1, dim)
                flat_tgt = target.contiguous().view(flat.shape[:-1])
                s = 0
                while s < flat.size(1):
                    e = s + self.softmax_batch
                    yield (flat[:, s:e],) + rest, flat_tgt[:, s:e], False
                    s = e

        def gather_target_probs(probs, target):
            probs = probs.gather(
                dim=2,
                index=target.unsqueeze(-1),
            )
            return probs

        orig_target = sample['target']

        # compute scores for each model in the ensemble
        stacked_lprobs = []
        target_lprobs = []
        avg_attn = None
        for model in models:
            model.eval()
            decoder_out = model(**net_input)
            attn = decoder_out[1]
            if type(attn) is dict:
                attn = attn.get('attn', None)

            batched = batch_for_softmax(decoder_out, orig_target)
            lprobs, idx = None, 0
            for bd, tgt, is_single in batched:
                sample['target'] = tgt
                curr_lprob = model.get_normalized_probs(bd, log_probs=True, sample=sample).data
                stacked_lprobs.append(curr_lprob)
                if is_single:
                    lprobs = gather_target_probs(curr_lprob, orig_target)
                else:
                    if lprobs is None:
                        lprobs = curr_lprob.new(orig_target.numel())
                    step = curr_lprob.size(0) * curr_lprob.size(1)
                    end = step + idx
                    tgt_lprobs = gather_target_probs(curr_lprob.view(tgt.shape + (curr_lprob.size(-1),)), tgt)
                    lprobs[idx:end] = tgt_lprobs.view(-1)
                    idx = end
                sample['target'] = orig_target

            lprobs = lprobs.view(sample['target'].shape)

            target_lprobs.append(lprobs)

            if attn is not None and torch.is_tensor(attn):
                attn = attn.data
                if avg_attn is None:
                    avg_attn = attn
                else:
                    avg_attn.add_(attn)

        if avg_attn is not None:
            avg_attn.div_(len(models))

        stacked_lprobs = torch.stack(stacked_lprobs, dim=0)
        esz = stacked_lprobs.size(0)
        #tok_unc = token_uncertainties(stacked_lprobs)

        target_lprobs = torch.stack(target_lprobs, dim=0)
        avg_lprobs = torch.logsumexp(target_lprobs, dim=0) - torch.log(torch.tensor
                                                                      (esz, dtype=torch.float32))
        aep_lprobs = target_lprobs.permute(1, 0, 2)
        stacked_lprobs = stacked_lprobs.permute(1,0,2,3)
        bsz = target_lprobs.size(1)
        hypos = []
        start_idxs = sample['start_indices'] if 'start_indices' in sample else [0] * bsz
        for i in range(bsz):
            # remove padding from ref
            ref = utils.strip_pad(sample['target'][i, start_idxs[i]:], self.pad) \
                if sample['target'] is not None else None
            tgt_len = ref.numel()
            avg_lprobs_i = avg_lprobs[i][start_idxs[i]:start_idxs[i] + tgt_len]

            score_i = avg_lprobs_i.sum() / tgt_len
            eos_enscores = torch.sum(aep_lprobs[i][:, :tgt_len], dim=1, keepdim=True)
            enscores = torch.cumsum(aep_lprobs[i][:, :tgt_len], dim=1)
            print(aep_lprobs.size(), eos_enscores.size(), stacked_lprobs.size(), tgt_len)
            tok_unc = token_uncertainties(stacked_lprobs[i][:, :tgt_len, :], enscores=enscores, step=tgt_len, decode=False)

            token_aep_tu = -avg_lprobs[i][start_idxs[i]:start_idxs[i] + tgt_len]
            token_aep_du = -torch.mean(aep_lprobs[i][:, :tgt_len], dim=0)
            print(token_aep_tu.size(), token_aep_du.size())
            aep_tu, aep_du, aep_nmpi = aep_uncertainty(eos_enscores, tgt_len-1)
            if avg_attn is not None:
                avg_attn_i = avg_attn[i]
                alignment = utils.extract_hard_alignment(
                    avg_attn_i,
                    sample['net_input']['src_tokens'][i],
                    sample['target'][i],
                    self.pad,
                    self.eos,
                )
            else:
                avg_attn_i = alignment = None
            hypos.append([{
                'tokens': ref,
                'score': score_i,
                'attention': avg_attn_i,
                'alignment': alignment,
                'positional_scores': avg_lprobs_i,
                'token_uncertainties': {
                    'entropy_of_expected': tok_unc['entropy_of_expected'][:tgt_len],
                    'expected_entropy': tok_unc['expected_entropy'][:tgt_len],
                    'mutual_information': tok_unc['mutual_information'][:tgt_len],
                    'EPKL': tok_unc['EPKL'][:tgt_len],
                    'MKL': tok_unc['MKL'][:tgt_len],
                    'ep_entropy_of_expected': tok_unc['ep_entropy_of_expected'][:tgt_len],
                    'ep_mutual_information': tok_unc['ep_mutual_information'][:tgt_len],
                    'ep_EPKL': tok_unc['ep_EPKL'][:tgt_len],
                    'ep_MKL': tok_unc['ep_MKL'][:tgt_len],
                    'token-aep-tu': token_aep_tu,
                    'token-aep-du': token_aep_du,
                    'token-aep-ku': token_aep_du - token_aep_tu
                },
                'sequence_uncertainties': {
                    'entropy_of_expected': torch.mean(tok_unc['entropy_of_expected'][:tgt_len]),
                    'expected_entropy': torch.mean(tok_unc['expected_entropy'][:tgt_len]),
                    'mutual_information': torch.mean(tok_unc['mutual_information'][:tgt_len]),
                    'EPKL': torch.mean(tok_unc['EPKL'][:tgt_len]),
                    'MKL': torch.mean(tok_unc['MKL'][:tgt_len]),
                    'ep_entropy_of_expected': torch.mean(tok_unc['ep_entropy_of_expected'][:tgt_len]),
                    'ep_mutual_information': torch.mean(tok_unc['ep_mutual_information'][:tgt_len]),
                    'ep_EPKL': torch.mean(tok_unc['ep_EPKL'][:tgt_len]),
                    'ep_MKL': torch.mean(tok_unc['ep_MKL'][:tgt_len]),
                    'score': -score_i,
                    'aep_tu': aep_tu.squeeze(),
                    'aep_du': aep_du.squeeze(),
                    'aep_npmi': aep_nmpi.squeeze(),
                    'score_npmi': aep_du.squeeze()+score_i,
                    'log-prob': score_i*tgt_len
                },
            }])
        return hypos
