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