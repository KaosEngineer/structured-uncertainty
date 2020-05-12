import torch


def token_uncertainties(stacked_lprobs, enscores, step, decode=True):
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
    elif step >0:
        if decode:
            scores = torch.logsumexp(enscores[:, :, step - 1].unsqueeze(-1), dim=0)
            nstacked_lprobs = stacked_lprobs + enscores[:, :, step - 1].unsqueeze(-1)
        else:
            nenscores = enscores.clone()
            nenscores[:,1:]=nenscores[:,:-1]
            nenscores[:,0]=0.0
            scores = torch.logsumexp(nenscores.unsqueeze(-1), dim=0)
            nstacked_lprobs = stacked_lprobs +nenscores.unsqueeze(-1)

        log_mprobs = torch.logsumexp(nstacked_lprobs, dim=0) - scores
        mprobs = torch.exp(log_mprobs)
        #print(torch.sum(mprobs,dim=1))
        #assert torch.all(torch.eq(torch.sum(mprobs,dim=1), torch.tensor(1, dtype=torch.float32))).item()
        expr_upper_bound = -torch.sum(mprobs * mlog_probs, dim=mdim)
        expr_eoe = -torch.sum(mprobs * log_mprobs, dim=mdim)
        expr_mkl = expr_upper_bound - expr_eoe
        expr_epkl = expr_upper_bound - exe
        expr_mutual_information = expr_eoe-exe

        #assert torch.all(torch.ge(expr_upper_bound, 0.0)).item()

    assert torch.all(torch.ge(expr_eoe, 0.0)).item()
    #assert torch.all(torch.ge(expr_mkl, 0.0)).item()
    #assert torch.all(torch.ge(expr_mutual_information, 0.0)).item()

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
    esz = torch.tensor(eos_enscores.size(0), dtype=torch.float32)
    prex_total_unc = -prex_eos_scores[:, step]
    step = torch.tensor(step, dtype=torch.float32)

    eoe_ub = -torch.mean(eos_enscores, dim=0)

    expr_total_unc = -(torch.logsumexp(eos_enscores, dim=0) - torch.log(esz))

    eoe_ub /= (step+1)
    prex_total_unc /= (step+1)
    expr_total_unc /= (step+1)

    expr_mkl = eoe_ub - expr_total_unc
    prex_mkl = eoe_ub - prex_total_unc

    return expr_total_unc, eoe_ub, expr_mkl, prex_total_unc, prex_mkl

def token_aep_uncertainty(pos_enscores):
    esz = torch.tensor(pos_enscores.size(0), dtype=torch.float32)
    eoe_ub = - torch.mean(pos_enscores, dim=0)
    prex_pos_scores = -(torch.logsumexp(pos_enscores, dim=0) -
                                           torch.log(esz))
    expr_scores = -(torch.logsumexp(torch.cumsum(pos_enscores, dim=2), dim=0) -
                                           torch.log(esz))
    expr_pos_scores = expr_scores.clone()
    expr_pos_scores[:,1:] = expr_scores[:, 1:] - expr_scores[:, :-1]

    prex_pos_mkl = eoe_ub - prex_pos_scores
    expr_pos_mkl = eoe_ub - expr_pos_scores
    return expr_pos_scores, eoe_ub, expr_pos_mkl, prex_pos_scores, prex_pos_mkl


EPS = 1e-8


def entropy(probs, dim=-1):
    return -(probs * (probs + EPS).log()).sum(dim=dim)


def compute_token_dirichlet_uncertainties(dirichlet_params, concentrations, expected_dirichlet):
    batch_size, num_tokens, vocab_size = dirichlet_params.size()

    entropy_of_expected = entropy(expected_dirichlet)
    expected_entropy = (-expected_dirichlet * (torch.digamma(dirichlet_params + 1) - torch.digamma(concentrations + 1))).sum(dim=-1)
    mutual_information = entropy_of_expected - expected_entropy

    epkl = (vocab_size - 1) / concentrations.squeeze(2)
    return entropy_of_expected, expected_entropy, mutual_information, epkl


def compute_sequence_dirichlet_uncertainties(dirichlet_params, concentrations, expected_logprobs, predict_inds, mask):
    log_probs = -(expected_logprobs.gather(-1, predict_inds.unsqueeze(-1)) * mask).sum(dim=1)
    scores = log_probs / mask.sum(dim=1)
    expected_scores = ((torch.digamma(dirichlet_params + 1) - torch.digamma(concentrations + 1)) * mask).sum(dim=1) / mask.sum(dim=1)
    expected_pmi = scores - expected_scores
    return log_probs, scores, expected_scores, expected_pmi
