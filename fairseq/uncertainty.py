import torch


def token_uncertainties(lprobs):
    dim = len(lprobs.size()) - 1
    esz = torch.tensor(lprobs.size()[0], dtype=torch.float32)
    probs = torch.exp(lprobs)
    exe = -torch.mean(torch.sum(probs * lprobs, dim=dim), dim=0)

    # Compute Entropy of Expected and Mutual Information
    mprobs = torch.mean(probs, dim=0)
    mdim = len(mprobs.size()) - 1

    log_mprobs = torch.logsumexp(lprobs, dim=0) - torch.log(esz)
    eoe = -torch.sum(mprobs * log_mprobs, dim=mdim)
    mutual_info = eoe - exe

    # Compute Expected Pairwise KL-divergence
    mlog_probs = torch.mean(lprobs, dim=0)
    eoe_upper_bound = -torch.sum(mprobs * mlog_probs, dim=mdim)
    epkl = eoe_upper_bound - exe

    return {'entropy_of_expected': eoe,
            'expected_entropy': exe,
            'mutual_information': mutual_info,
            'EPKL': epkl}

def aep_uncertainty(eos_enscores, step):
    esz = torch.tensor(eos_enscores.size(0), dtype=torch.float32)
    step = torch.tensor(step, dtype=torch.float32)

    data_unc = -torch.mean(eos_enscores, dim=0)
    total_unc = -(torch.logsumexp(eos_enscores, dim=0) - torch.log(esz))

    data_unc /= (step+1)
    total_unc /= (step+1)

    npmi = data_unc - total_unc

    return total_unc, data_unc, npmi


def token_aep_uncertainty(pos_enscores):
    esz = torch.tensor(pos_enscores.size(0), dtype=torch.float32)
    data_unc = - torch.mean(pos_enscores, dim=0)

    total_unc = -(torch.logsumexp(pos_enscores, dim=0) - torch.log(esz))
    know_unc = data_unc-total_unc

    return total_unc, data_unc, know_unc
