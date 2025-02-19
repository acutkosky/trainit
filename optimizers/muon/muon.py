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

from typing import NamedTuple
from jaxtyping import Array, PyTree

import warnings

from utils import tree_utils
from optimizers.base import adamw
from optimizers.combine import multi_transform
from optimizers.schedule import get_current_lr
from optimizers.muon.base import newton_schulz, LabelParamsFn, scale_by_offset
from optimizers.muon.mango import normalize_with_grad_squared


# =============================================================================
# Label functions to partition GPT model.
# =============================================================================

def muon_label_params_default(params):
    """Default way to partition layers.
    
    Following the default implementation from (Jordan, 2024)

    https://github.com/KellerJordan/Muon/blob/28c793b55ef1cf86e5d6091bfbdbe0029b11eabb/muon.py#L34

    Parses 1d arrays and embedding, head layers to adamw; and all other 2d arrays as muon.
    """
    def parse_path(path, p):
        parts = [part.name for part in path if isinstance(part, jtu.GetAttrKey)]
        muon_label = "muon"
        adamw_label = "adamw"
        # Detect embedding layers and head layers
        if "token_embedding" in parts or "position_embedding" in parts:
            return adamw_label
        if "head" in parts:
            return adamw_label
        if p.ndim == 1:
            return adamw_label
        if p.ndim == 2:
            return muon_label
        raise ValueError(f"cannot categorize parameter: {p}")
    return jtu.tree_map_with_path(parse_path, params) 


def muon_label_params_by_dimension(params):
    """A simplified label function that labels 2d arrays as muon and 1d arrays as adamw."""
    return jtu.tree_map(
        lambda p: "muon" if p.ndim == 2 else "adamw", params
    )


# =============================================================================
# Main scale_by_muon and muon function.
# =============================================================================

class ScaleByMuonState(NamedTuple):
    """scale_by_muon state."""
    count: Array
    muon_momentum: optax.Updates


def scale_by_muon(
        learning_rate: optax.ScalarOrSchedule = 0.05,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 6,
) -> optax.GradientTransformation:
    """Muon update on parameters with 2d arrays."""
    
    def init_fn(params):
        return ScaleByMuonState(
            count = jnp.zeros([], dtype=jnp.int32),
            muon_momentum = tree_utils.zeros_like(params),
        )
    
    def update_fn(updates, state, params=None):
        del params
        count = state.count
        muon_momentum = state.muon_momentum

        # Update momentum.
        muon_momentum = jtu.tree_map(
            lambda mu, g: momentum * mu + g, muon_momentum, updates)

        # Apply nesterov's momentum before applying normalization.
        if nesterov:
            updates = jtu.tree_map(
                lambda mu, g: momentum * mu + g, muon_momentum, updates)
        else:
            updates = muon_momentum

        # Orthogonalize momentum matrix.
        updates = jtu.tree_map(
            lambda G: newton_schulz(G, steps=ns_steps), updates)
        
        # Additional scaling based on shape (see line 135).
        updates = jtu.tree_map(
            lambda G: G * max(1, G.shape[0]/G.shape[1])**0.5, updates)

        # Wrap final update.
        lr = get_current_lr(learning_rate, count)
        updates = tree_utils.scalar_dot(updates, -lr)

        return updates, ScaleByMuonState(
            count = optax.safe_int32_increment(count),
            muon_momentum = muon_momentum
        )
    
    return optax.GradientTransformation(init_fn, update_fn)


def muon(
        learning_rate: optax.ScalarOrSchedule = 0.05,
        momentum: float = 0.95,
        nesterov: bool = True,
        beta2: float = 0.0,
        p_pre: float = 0.5,
        p_post: float = 0.0,
        offset_beta: float = 0.0,
        ns_steps: int = 6,
        adam_lr: optax.ScalarOrSchedule = 3e-4,
        adam_beta1: float = 0.95,
        adam_beta2: float = 0.95,
        adam_eps: float = 1e-8,
        adam_wd: float = 0.0,
        label_params: LabelParamsFn | None = None
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
    if label_params is None:
        label_params = muon_label_params_default    # use default muon partition.
    if not beta2:
        optim_muon = scale_by_muon(
            learning_rate, momentum, nesterov, ns_steps
        )
    else:
        def normalize(G):
            G = newton_schulz(G, steps=ns_steps)
            G = G * max(1, G.shape[0]/G.shape[1])**0.5
        optim_muon = optax.trace(decay=momentum, nesterov=nesterov)
        optim_muon = normalize_with_grad_squared(
            inner=optim_muon,
            normalize_fn=normalize,
            beta=beta2,
            eps=1e-8, 
            power_pre=p_pre,
            power_post=p_post,
            correct_bias=True,
        )
        optim_muon = optax.chain(
            optim_muon,
            optax.scale_by_learning_rate(learning_rate)
        )
    optim_adam = adamw(
        learning_rate=adam_lr,
        beta1=adam_beta1,
        beta2=adam_beta2,
        eps=adam_eps,
        weight_decay=adam_wd,
        use_nesterov=False,
    )
    # NOTE: we could switch to optax implementation if needed.
    # optim_adam = optax.adamw(
    #     learning_rate=adam_lr,
    #     b1=adam_beta1,
    #     b2=adam_beta2,
    #     eps=adam_eps,
    #     weight_decay=adam_wd,
    #     nesterov=False,
    # )
    transforms = {
        "muon": optim_muon,
        "adamw": optim_adam,
    }
    optim = multi_transform(transforms, label_params)
    if offset_beta:
        optim = optax.chain(
            optim,
            scale_by_offset(offset_beta),
        )
    return optim


def muon_og(
        learning_rate: optax.ScalarOrSchedule = 0.05,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 6,
        ns_embedding: bool = False,
        ns_head: bool = False,
        adam_lr: optax.ScalarOrSchedule = 3e-4,
        adam_beta1: float = 0.95,
        adam_beta2: float = 0.95,
        adam_eps: float = 1e-8,
        adam_wd: float = 0.0,
) -> optax.GradientTransformation:
    """The OG muon optimizer that doesn't apply newton-schulz on embedding nor head layers."""
    warnings.warn("muon_og is deprecated. please use muon instead.",
                  "define new label functions if necessary.")
    
    optim_muon = scale_by_muon(
        learning_rate, momentum, nesterov, ns_steps
    )
    optim_momentum = optax.chain(
        optax.trace(decay=momentum, nesterov=nesterov),
        optax.scale_by_learning_rate(learning_rate)
    )
    optim_adamw = adamw(
        learning_rate=adam_lr,
        beta1=adam_beta1,
        beta2=adam_beta2,
        eps=adam_eps,
        weight_decay=adam_wd,
        use_nesterov=False,
    )
    transforms = {
        "muon": optim_muon,
        "momentum": optim_momentum,
        "adamw": optim_adamw,
    }
    def label_params(params):
        def get_layer(path, p):
            parts = [part.name for part in path if isinstance(part, jtu.GetAttrKey)]
            # Special ararys.
            if "token_embedding" in parts or "position_embedding" in parts:
                return "embedding"
            if "head" in parts:
                return "head"
            # General arrays.
            if p.ndim == 2:
                return "mat"
            if p.ndim == 1:
                return "vec"
            raise ValueError(f"cannot categorize parameter: {p}")
        parse_table = {
            # NOTE: we change to use adamw instead of SGDM for all layers which muon is not applied
            # "embedding": "muon" if ns_embedding else "momentum",
            # "head": "muon" if ns_head else "momentum",
            "embedding": "muon" if ns_embedding else "adamw",
            "head": "muon" if ns_head else "adamw",
            "mat": "muon",
            "vec": "adamw",
        }
        def fn(path, p):
            return parse_table[get_layer(path, p)]
        return jtu.tree_map_with_path(fn, params) 
    return multi_transform(transforms, label_params)
    