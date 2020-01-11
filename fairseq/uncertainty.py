import torch

# Always expects models to be in dim=0 and vocab to be in last dimension


def entropy_of_expected(probs, epsilon=1e-12):
    probs = torch.mean(probs, dim=0)
    log_probs = -torch.log(probs + epsilon)
    return torch.sum(probs * log_probs, dim=probs.size(-1))


def expected_entropy(probs, epsilon=1e-12):
    log_probs = -torch.log(probs + epsilon)
    return torch.mean(torch.sum(probs * log_probs, dim=probs.size(-1)), dim=0)


def mutual_information(probs, epsilon):
    eoe = entropy_of_expected(probs, epsilon)
    exe = expected_entropy(probs, epsilon)
    return eoe - exe


def expected_pairwise_kl_divergence(probs, epsilon=1e-12):
    exe = entropy_of_expected(probs, epsilon)
    log_probs = -torch.mean(torch.log(probs + epsilon), dim=0)
    probs = torch.mean(probs, dim=0)
    eoe_upper_bound = torch.sum(probs * log_probs, dim=probs.size(-1))
    return eoe_upper_bound - exe


def token_uncertainties(probs, epsilon=1e-12):
    #Compute Expected Entropy
    log_probs = -torch.log(probs + epsilon)
    exe = torch.mean(torch.sum(probs*log_probs, dim=probs.size(-1)), dim=0)

    #Compute Entropy of Expected and Mutual Information
    mprobs = torch.mean(probs, dim=0)
    log_mprobs = -torch.log(mprobs+epsilon)
    eoe = torch.sum(mprobs*log_mprobs, dim=mprobs.size(-1))
    mutual_info = eoe - exe

    #Compute Expected Pairwise KL-divergence
    mlog_probs = torch.mean(log_probs, dim=0)
    eoe_upper_bound = torch.sum(mprobs*mlog_probs, dim=mprobs.size(-1))
    epkl = eoe_upper_bound - exe

    uncertainty = {#'confidence': conf,
                   'entropy_of_expected': eoe,
                   'expected_entropy': exe,
                   'mutual_information': mutual_info,
                   'EPKL': epkl}

    return uncertainty


def sequence_uncertainties(probs, epsilon=1e-12):
    """
    :param probs: torch tensor of dimensions [n_models, batch_size, seq_len, vocab_size]
    :param epsilon: a smoothing constant
    :return: a dictionary of uncertainty measures, each a tensor [batch_size, seq_len]
    """

    token_uncertainty = token_uncertainties(probs, epsilon)

    sequence_uncertainty = {#'confidence': conf,
                            'entropy_of_expected': torch.mean(token_uncertainty['entropy_of_expected'], dim=1),
                            'expected_entropy': torch.mean(token_uncertainty['expected_entropy'], dim=1),
                            'mutual_information': torch.mean(token_uncertainty['mutual_information'], dim=1),
                            'EPKL': torch.mean(token_uncertainty['EPKL'], dim=1)}

    return sequence_uncertainty, token_uncertainty
