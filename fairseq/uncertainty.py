import torch

EPS = 1e-10


def token_uncertainties(stacked_lprobs, enscores, step, decode=True):
    """

    :param stacked_lprobs:
    :param enscores:
    :param step:
    :param decode:
    :return:
    """
    dim = len(stacked_lprobs.size()) - 1
    esz = torch.tensor(stacked_lprobs.size()[0], dtype=torch.float32)
    probs = torch.exp(stacked_lprobs)
    exe = -torch.mean(torch.sum(probs * stacked_lprobs, dim=dim), dim=0)

    # Compute Entropy of Expected and Mutual Information
    mprobs = torch.mean(probs, dim=0)
    mdim = len(mprobs.size()) - 1

    log_mprobs = torch.logsumexp(stacked_lprobs, dim=0) - torch.log(esz)
    eoe = -torch.sum(mprobs * log_mprobs, dim=mdim)
    mutual_info = eoe - exe

    # Compute Expected Pairwise KL-divergence
    mlog_probs = torch.mean(stacked_lprobs, dim=0)
    eoe_upper_bound = -torch.sum(mprobs * mlog_probs, dim=mdim)
    epkl = eoe_upper_bound - exe
    mkl = eoe_upper_bound - eoe

    if step == 0:
        expr_eoe = eoe
        expr_mkl = mkl
        expr_epkl = epkl
        expr_mutual_information = mutual_info
    elif step > 0:
        if decode:
            scores = torch.logsumexp(enscores[:, :, step - 1].unsqueeze(-1), dim=0)
            nstacked_lprobs = stacked_lprobs + enscores[:, :, step - 1].unsqueeze(-1)
        else:
            nenscores = enscores.clone()
            nenscores[:, 1:] = nenscores[:, :-1]
            nenscores[:, 0] = 0.0
            scores = torch.logsumexp(nenscores.unsqueeze(-1), dim=0)
            nstacked_lprobs = stacked_lprobs + nenscores.unsqueeze(-1)

        log_mprobs = torch.logsumexp(nstacked_lprobs, dim=0) - scores
        mprobs = torch.exp(log_mprobs)
        # print(torch.sum(mprobs,dim=1))
        # assert torch.all(torch.eq(torch.sum(mprobs,dim=1), torch.tensor(1, dtype=torch.float32))).item()
        expr_upper_bound = -torch.sum(mprobs * mlog_probs, dim=mdim)
        expr_eoe = -torch.sum(mprobs * log_mprobs, dim=mdim)
        expr_mkl = expr_upper_bound - expr_eoe
        expr_epkl = expr_upper_bound - exe
        expr_mutual_information = expr_eoe - exe

        # assert torch.all(torch.ge(expr_upper_bound, 0.0)).item()

    assert torch.all(torch.ge(expr_eoe, 0.0)).item()
    # assert torch.all(torch.ge(expr_mkl, 0.0)).item()
    # assert torch.all(torch.ge(expr_mutual_information, 0.0)).item()

    return {'entropy_of_expected': eoe,
            'ep_entropy_of_expected': expr_eoe,
            'expected_entropy': exe,
            'mutual_information': mutual_info,
            'ep_mutual_information': expr_mutual_information,
            'EPKL': epkl,
            'ep_EPKL': expr_epkl,
            'MKL': mkl,
            'ep_MKL': expr_mkl}


def seq_uncertainties(eos_enscores, prex_eos_scores, step):
    """

    :param eos_enscores:
    :param prex_eos_scores:
    :param step:
    :return:
    """
    esz = torch.tensor(eos_enscores.size(0), dtype=torch.float32)
    prex_total_unc = -prex_eos_scores[:, step]
    step = torch.tensor(step, dtype=torch.float32)

    eoe_ub = -torch.mean(eos_enscores, dim=0)
    expr_total_unc = -(torch.logsumexp(eos_enscores, dim=0) - torch.log(esz))

    expr_var = torch.var(torch.exp(eos_enscores / (step + 1)) + EPS, dim=0, unbiased=False)
    expr_logvar = torch.var(eos_enscores / (step + 1) + EPS, dim=0, unbiased=False)
    expr_varcombo = -(1.0 - expr_var / (EPS + torch.exp(-expr_total_unc / (step + 1))))
    expr_logcombo = -(1 - expr_logvar / (EPS + torch.mean(eos_enscores / (step + 1), dim=0)))

    eoe_ub /= (step + 1)
    prex_total_unc /= (step + 1)
    expr_total_unc /= (step + 1)

    expr_mkl = eoe_ub - expr_total_unc
    prex_mkl = eoe_ub - prex_total_unc

    return expr_total_unc, eoe_ub, expr_mkl, prex_total_unc, prex_mkl, expr_var, expr_varcombo, expr_logvar, expr_logcombo


def token_aep_uncertainty(pos_enscores):
    """

    :param pos_enscores:
    :return:
    """
    esz = torch.tensor(pos_enscores.size(0), dtype=torch.float32)
    eoe_ub = - torch.mean(pos_enscores, dim=0)
    prex_pos_scores = -(torch.logsumexp(pos_enscores, dim=0) - torch.log(esz))
    expr_scores = -(torch.logsumexp(torch.cumsum(pos_enscores, dim=2), dim=0) - torch.log(esz))
    expr_pos_scores = expr_scores.clone()
    expr_pos_scores[:, 1:] = expr_scores[:, 1:] - expr_scores[:, :-1]

    prex_pos_mkl = eoe_ub - prex_pos_scores
    expr_pos_mkl = eoe_ub - expr_pos_scores
    return expr_pos_scores, eoe_ub, expr_pos_mkl, prex_pos_scores, prex_pos_mkl


def entropy(probs, dim: int = -1):
    return -(probs * (probs + EPS).log()).sum(dim=dim)


def compute_token_dirichlet_uncertainties(dirichlet_params, precisions, expected_dirichlet):
    """
    Function which computes token-level measures of uncertainty for Dirichlet model.

    :param dirichlet_params:  Tensor of size [batch_size, seq_len, vocab_size] of Dirichlet concentration parameters.
    :param precisions: Tensor of size [batch_size, seq_len, 1] of Dirichlet Precisions
    :param expected_dirichlet: Tensor of size [batch_size, seq_len, vocab_size] of probablities of expected categorical under Dirichlet.
    :return: Tensors of token level uncertainties of size [batch_size, seq_len]
    """
    batch_size, num_tokens, vocab_size = dirichlet_params.size()

    entropy_of_expected = entropy(expected_dirichlet)
    assert (entropy_of_expected >= 0).all()
    expected_entropy = (
            -expected_dirichlet * (torch.digamma(dirichlet_params + 1) - torch.digamma(precisions + 1))).sum(
        dim=-1)
    assert (expected_entropy >= -1e-3).all()

    mutual_information = entropy_of_expected - expected_entropy
    assert (mutual_information >= -1e-3).all()
    epkl = (vocab_size - 1) / precisions.squeeze(2)
    assert (epkl >= 0).all()
    mkl = (-expected_dirichlet * (torch.digamma(dirichlet_params + EPS) - torch.digamma(precisions + EPS))).sum(
        dim=-1) - entropy_of_expected
    assert (mkl >= 0).all()

    return entropy_of_expected, expected_entropy, mutual_information, epkl, mkl


def compute_sequence_dirichlet_uncertainties(dirichlet_params, precisions, log_expected_probs, predict_inds, mask,
                                             num_tokens):
    """

    :param dirichlet_params:  Tensor of size [batch_size, seq_len, vocab_size] of Dirichlet concentration parameters.
    :param precisions: Tensor of size [batch_size, seq_len, 1] of Dirichlet Precisions
    :param log_expected_probs:  Tensor of size [batch_size, seq_len, vocab_size] of log-probablities of expected categorical under Dirichlet.
    :param predict_inds: Tensor of size [batch_size, seq_len] of token ids
    :param mask:  Tensor of size [batch_size, seq_len] of masked token ids
    :param num_tokens:  Tensor of size [batch_size] of masked token ids
    :return:
    """
    unsqueezed_inds = predict_inds.unsqueeze(-1) # now [batch_size, seq_len, 1]

    token_log_probs = log_expected_probs.gather(-1, unsqueezed_inds).squeeze(2)
    # token_log_probs now [batch_size, seq_len]
    if mask.any():
        token_log_probs.masked_fill(mask, 0)

    log_probs = token_log_probs.sum(dim=1)
    scores = -log_probs / num_tokens
    # scores >=0

    token_scores_mkl = (torch.digamma(precisions + EPS) - torch.digamma(
        dirichlet_params.gather(-1, unsqueezed_inds) + EPS)
                        ).squeeze(2) + token_log_probs

    if mask.any():
        token_scores_mkl.masked_fill(mask, 0)

    scores_mkl = token_scores_mkl.sum(dim=1) / num_tokens
    return log_probs, scores, scores_mkl
