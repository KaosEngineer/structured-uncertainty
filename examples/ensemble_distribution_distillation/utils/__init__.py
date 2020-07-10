import torch


def freeze_module_params(m):
    if m is not None:
        for p in m.parameters():
            p.requires_grad = False


@torch.jit.script
def exp_with_max_subtraction(x):
    """The way they actually do it in softmax"""
    max_values, max_indices = x.max(dim=-1, keepdim=True)
    x_sub_max = x - max_values
    return torch.exp(x_sub_max)


prob_parametrization = {
    'exp': torch.exp,
    'softplus': torch.nn.functional.softplus,
    'exp_max_sub': exp_with_max_subtraction,
}


def get_dirichlet_parameters(model, net_output, parametrization_func, add_to_alphas=0):
    logits, extra = net_output

    if 'dirichlet_params' in extra:
        means = model.get_normalized_probs(net_output, log_probs=False)
        precision = parametrization_func(extra['dirichlet_params'])
        alphas = means * precision + add_to_alphas
        precision = precision.squeeze(2) + add_to_alphas * alphas.size(-1)
    else:
        alphas = parametrization_func(logits) + add_to_alphas
        precision = torch.sum(alphas, dim=-1)
    return alphas, precision
