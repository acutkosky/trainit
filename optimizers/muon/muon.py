# An optax implementation of muon optimizer from:
# 
# https://github.com/KellerJordan/Muon
# 
# This implementation adapts the pytorch optimizer
# to an optax.GradientTransformation object.

"""Muon optimizer."""

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optax

from typing import NamedTuple, Optional
from jaxtyping import Array, PyTree

from utils import tree_utils
from optimizers.base import adamw
from optimizers.combine import multi_transform
from optimizers.schedule import get_current_lr


def newton_schulz(G: Array, steps: int) -> Array:
    """An approximate Newton-Schulz method.
    
    Adapted from:

    https://github.com/KellerJordan/Muon/blob/master/muon.py

    Given a matrix G with SVD decomposition SUV^T, this function
    approximates US'V^T where S' is diagonal with values Uniform(0.5, 1.5)
    without needing to compute any SVD decomposition.
    """
    assert G.ndim == 2

    # NOTE: do we need to adapt to bfloat16 as in the OG repo?
    a, b, c = (3.4445, -4.7750,  2.0315)
    eps = 1e-7

    X = G
    if G.shape[0] > G.shape[1]:
        X = X.T

    # wrap iterative update with jax.lax.fori_loop.
    X /= (jnp.linalg.norm(X, ord=2) + eps)
    def body_func(i, val):
        X = val
        A = X @ X.T
        B = b * A + c * A @ A
        return a * X + B @ X
    X = jax.lax.fori_loop(
        0, steps, body_func, X
    )

    if G.shape[0] > G.shape[1]:
        X = X.T
    return X


class ScaleByMuonState(NamedTuple):
    """scale_by_muon state."""
    count: Array
    muon_momentum: optax.Updates
    grad_squared: optax.Updates
    offset: optax.Updates


def scale_by_muon(
        learning_rate: optax.ScalarOrSchedule = 0.05,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 6,
        use_l2: bool = False,
        offset_beta: Optional[float] = None,
        beta2: Optional[float]=None,
) -> optax.GradientTransformation:
    """Muon update on parameters with 2d arrays."""
    
    def init_fn(params):
        return ScaleByMuonState(
            count = jnp.zeros([], dtype=jnp.int32),
            muon_momentum = tree_utils.zeros_like(params),
            grad_squared = tree_utils.zeros_like(params) if beta2 else None,
            offset = tree_utils.zeros_like(params),
        )
    
    def update_fn(updates, state, params=None):
        del params
        count = state.count
        muon_momentum = state.muon_momentum

        if beta2:
            next_grad_squared = jtu.tree_map(
                lambda v, g: v * beta2 + g**2,
                state.grad_squared,
                updates
            )
            updates = jtu.tree_map(
                lambda g, v: g/(jnp.sqrt(v) + 1e-8),
                updates,
                next_grad_squared
            )
        else:
            next_grad_squared = state.grad_squared

        # Update momentum.
        muon_momentum = jtu.tree_map(
            lambda mu, g: momentum * mu + g, muon_momentum, updates)

        

        # Orthogonalize momentum matrix.
        if nesterov:
            # Apply nesterov's momentum before applying normalization.
            updates = jtu.tree_map(
                lambda mu, g: momentum * mu + g, muon_momentum, updates)
        else:
            updates = muon_momentum
        if use_l2:
            updates = jtu.tree_map(
                lambda G: G/(jnp.linalg.norm(G) + 1e-8),
                updates
            )
        else:
            updates = jtu.tree_map(
                lambda G: newton_schulz(G, steps=ns_steps), updates)
        
        # Additional scaling based on shape (see line 135).
        updates = jtu.tree_map(
            lambda G: G * max(1, G.shape[0]/G.shape[1])**0.5, updates)

        if offset_beta:
            next_offset = jtu.tree_map(
                lambda o, u: o * offset_beta + (1-offset_beta) * u,
                state.offset,
                updates
            )

            updates = jtu.tree_map(
                lambda a, b: a+b,
                updates,
                next_offset
            )
        else:
            next_offset = state.offset

        # Wrap final update.
        lr = get_current_lr(learning_rate, count)
        updates = tree_utils.scalar_dot(updates, -lr)

        return updates, ScaleByMuonState(
            count = optax.safe_int32_increment(count),
            muon_momentum = muon_momentum,
            offset=state.offset,
            grad_squared=next_grad_squared
        )
    
    return optax.GradientTransformation(init_fn, update_fn)


def muon(
        learning_rate: optax.ScalarOrSchedule = 0.05,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 6,
        use_l2: bool = False,
        adam_lr: optax.ScalarOrSchedule = 3e-4,
        adam_beta1: float = 0.95,
        adam_beta2: float = 0.95,
        adam_eps: float = 1e-8,
        adam_wd: float = 0.0,
        offset_beta: float = 1.0,
        beta2: Optional[float]=None,
) -> optax.GradientTransformation:
    """The muon optimizer.
    
    Applies muon update on suitable parameters and
    applies adam update on the rest.

    We use `optax.multi_transform` to combine these updates.

    Args:
        learning_rate: muon learning rate.
        momentum: sgd momentum of muon.
        nesterov: whether to use nesterov momentum.
        ns_steps: number of steps of Newton-Schulz.
        adam_lr: adam learning rate.
        adam_beta1: adam beta1.
        adam_beta2: adam beta2.
        adam_eps: adam eps.
        adam_wd: adam weight decay.
    """
    optim_muon = scale_by_muon(
        learning_rate, momentum, nesterov, ns_steps, use_l2, offset_beta, beta2=beta2
    )
    optim_adam = adamw(
        learning_rate=adam_lr,
        beta1=adam_beta1,
        beta2=adam_beta2,
        eps=adam_eps,
        weight_decay=adam_wd,
        use_nesterov=False,
        offset_beta=offset_beta,
    )
    transforms = {
        "muon": optim_muon,
        "adam": optim_adam,
    }
    def label_params(params):
        return jtu.tree_map(
            lambda p: "muon" if p.ndim == 2 else "adam", params
        )
    return multi_transform(transforms, label_params)
