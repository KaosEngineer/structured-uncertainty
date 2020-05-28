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
        expr_upper_bound = -torch.sum(mprobs * mlog_probs, dim=mdim)
        expr_eoe = -torch.sum(mprobs * log_mprobs, dim=mdim)
        expr_mkl = expr_upper_bound - expr_eoe
        expr_epkl = expr_upper_bound - exe
        expr_mutual_information = expr_eoe-exe

    return {'entropy_of_expected': eoe,
            'ep_entropy_of_expected': expr_eoe,
            'expected_entropy': exe,
            'mutual_information': mutual_info,
            'ep_mutual_information': expr_mutual_information,
            'EPKL': epkl,
            'ep_EPKL': expr_epkl,
            'MKL': mkl,
            'ep_MKL': expr_mkl}

def seq_uncertainties(eos_enscores, step):
    esz = torch.tensor(eos_enscores.size(0), dtype=torch.float32)
    step = torch.tensor(step, dtype=torch.float32)

    eoe_ub = -torch.mean(eos_enscores, dim=0)
    total_unc = -(torch.logsumexp(eos_enscores, dim=0) - torch.log(esz))

    eoe_ub /= (step+1)
    total_unc /= (step+1)

    mkl = eoe_ub - total_unc

    return total_unc, eoe_ub, mkl


def token_aep_uncertainty(pos_enscores):
    esz = torch.tensor(pos_enscores.size(0), dtype=torch.float32)
    eoe_ub = - torch.mean(pos_enscores, dim=0)

    total_unc = -(torch.logsumexp(pos_enscores, dim=0) - torch.log(esz))
    know_unc = eoe_ub-total_unc

    return total_unc, eoe_ub, know_unc
