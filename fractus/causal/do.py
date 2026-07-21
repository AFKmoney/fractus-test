"""do_intervention: atomic intervention operator (Pearl's do-operator).

do(X_i = v) fixes X_i to value v for all samples. This is the primitive
building block of Pearl's causal framework. Full do-calculus (backdoor /
front-door identification rules) is not implemented here.

Differentiable (for estimating causal effects via gradient when the model
is differentiable).
"""

import torch


def do_intervention(x, var_idx, value):
    """Apply do(X_{var_idx} = value) to a batch of data.

    Args:
        x:       tensor (N, d) of observed variables.
        var_idx: index of the variable to intervene on.
        value:   value to impose (can be non-zero).
    Returns:
        x_intervened: (N, d) with column var_idx set to `value`.
    """
    x_intervened = x.clone()
    x_intervened[:, var_idx] = value
    return x_intervened
