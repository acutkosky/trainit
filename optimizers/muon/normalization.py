"""Normalization methods on jnp.Arrays."""

import jax
import jax.numpy as jnp
from typing import NamedTuple, Optional, Callable, Literal, Dict, Tuple
from jaxtyping import Array
from optimizers.muon.base import newton_schulz


ArrayFn = Callable[[Array], Array]


def split_vmap(f: ArrayFn, num_heads: int = 1) -> ArrayFn:
    """Broadcasts a function f: [d,:] -> [d,:] to a matrix/vector [3nd,:]
    in a way that first reshapes into [3,n,d,:], then applies f on [d,:],
    and finally reshape back into [3nd,:].
    """
    def split_fn(G):
        assert G.ndim == 1 or G.ndim == 2
        assert G.shape[0] % (3 * num_heads) == 0
        ndim = G.ndim
        shape = G.shape
        n = num_heads
        d = G.shape[0] // (3 * num_heads)
        # Reshape into [3,n,d,:].
        if ndim == 1:
            G = jnp.reshape(G, (3, n, d))
        else:
            G = jnp.reshape(G, (3, n, d, shape[1]))
        # Use nested vmap to broadcast mapping f to the last axes (d,:).
        G = jax.vmap(jax.vmap(f))(G)
        # Reshape back into [3nd,:].
        if ndim == 1:
            G = jnp.reshape(G, (3*n*d,))
        else:
            G = jnp.reshape(G, (3*n*d, shape[1]))
        return G
    return split_fn


def get_normalize_fn(
        normalize: str | None = None,
        eps: float = 1e-8,
        ns_steps: int = 6,
        num_heads: int = 12,
        scale_dim: bool = False,
        transpose: bool = False,
        clip_min: float | None = None,
        clip_max: float | None = None,
) -> ArrayFn:
    if clip_min is None:
        clip_min = 0.0
    if clip_max is None:
        clip_max = jnp.inf
    clip = lambda x: jnp.clip(x, min=clip_min, max=clip_max)

    # Base normalization functions.
    identity = lambda G: G

    normalize_l2 = lambda G: G / (jnp.linalg.norm(G) + eps)
    scale_l2 = lambda G: G * clip(len(G)**0.5)

    normalize_l2_row = lambda G: G / (jnp.linalg.norm(G, axis=1, keepdims=True) + eps)
    scale_l2_row = lambda G: G * clip((1/G.shape[1])**0.5)

    normalize_l2_col = lambda G: G / (jnp.linalg.norm(G, axis=0, keepdims=True) + eps)
    scale_l2_col = lambda G: G * clip(G.shape[0]**0.5)

    normalize_ns = lambda G: newton_schulz(G, steps=ns_steps)
    scale_ns = lambda G: G * clip((G.shape[0]/G.shape[1])**0.5)

    normalize_inf = jnp.sign

    def wrap_normalize_l2(G):
        assert G.ndim == 1
        G = normalize_l2(G)
        if scale_dim:
            G = scale_l2(G)
        return G

    def wrap_normalize_l2_row(G):
        assert G.ndim == 2
        if transpose:
            G = jnp.transpose(G)
        G = normalize_l2_row(G)
        if scale_dim:
            G = scale_l2_row(G)
        if transpose:
            G = jnp.transpose(G)
        return G
    
    def wrap_normalize_l2_col(G):
        assert G.ndim == 2
        if transpose:
            G = jnp.transpose(G)
        G = normalize_l2_col(G)
        if scale_dim:
            G = scale_l2_col(G)
        if transpose:
            G = jnp.transpose(G)
        return G
    
    wrap_normalize_l2_split = split_vmap(wrap_normalize_l2, num_heads=num_heads)

    wrap_normalize_inf = normalize_inf

    def wrap_normalize_ns(G):
        assert G.ndim == 2
        G = normalize_ns(G)
        if scale_dim:
            G = scale_ns(G)
        return G
    
    wrap_normalize_ns_split = split_vmap(wrap_normalize_ns, num_heads=num_heads)

    # Assign normalize functions.
    if normalize is None:
        normalize_fn = identity
    elif normalize == "l2":
        normalize_fn = wrap_normalize_l2
    elif normalize == "l2_row":
        normalize_fn = wrap_normalize_l2_row
    elif normalize == "l2_col":
        normalize_fn = wrap_normalize_l2_col
    elif normalize == "l2_split":
        normalize_fn = wrap_normalize_l2_split
    elif normalize == "inf_":
        normalize_fn = wrap_normalize_inf
    elif normalize == "ns":
        normalize_fn = wrap_normalize_ns
    elif normalize == "ns_split":
        normalize_fn = wrap_normalize_ns_split
    else:
        raise ValueError(f"invalid normalization type = '{normalize}'.")
    
    return normalize_fn