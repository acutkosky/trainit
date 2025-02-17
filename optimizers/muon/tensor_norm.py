"""Normalization methods on jnp.Arrays."""

import jax
import jax.numpy as jnp
from typing import NamedTuple, Optional, Callable, Literal, Dict, Tuple
from jaxtyping import Array


"""Normalization methods on jnp.Arrays."""

import jax
import jax.numpy as jnp
from typing import Callable
from jaxtyping import Array


ArrayFn = Callable[[Array], float]


# -----------------------------------------------------------------------------
# Base tensor norm functions for 1D arrays
# -----------------------------------------------------------------------------

def _norm_l2(G):
    """L2 norm (1D)."""
    assert G.ndim == 1
    return jnp.linalg.norm(G)

def _norm_l1(G):
    """L1 norm (1D)."""
    assert G.ndim == 1
    return jnp.linalg.norm(G, ord=1)

def _norm_inf(G):
    """Infinity norm (1D)."""
    assert G.ndim == 1
    return jnp.linalg.norm(G, ord=jnp.inf)

def _norm_rms(G):
    """RMS norm (1D)."""
    assert G.ndim == 1
    return jnp.linalg.norm(G) / (G.shape[0] ** 0.5)


# -----------------------------------------------------------------------------
# Base tensor norm functions for 2D arrays
# -----------------------------------------------------------------------------

def _norm_op(G):
    """Operator norm (2D)."""
    assert G.ndim == 2
    return jnp.linalg.norm(G, ord=2)

def _norm_fro(G):
    """Frobenius norm (2D)."""
    assert G.ndim == 2
    return jnp.linalg.norm(G)

def _norm_colmax_l2(G):
    """Max column L2 norm (2D)."""
    assert G.ndim == 2
    return jnp.max(jnp.linalg.norm(G, axis=0))

def _norm_rowmax_l2(G):
    """Max row L2 norm (2D)."""
    assert G.ndim == 2
    return jnp.max(jnp.linalg.norm(G, axis=1))

def _norm_max(G):
    """Matrix max norm (2D)."""
    assert G.ndim == 2
    return jnp.max(jnp.abs(G))


# -----------------------------------------------------------------------------
# Induced norms for 1D arrays
# -----------------------------------------------------------------------------

def _induced_norm_vec_l2_l2(G):
    """Inf norm (1D)."""
    return _norm_inf(G)

def _induced_norm_vec_l2_rms(G):
    """sqrt(1/d) * inf norm (1D)."""
    return _norm_inf(G) * (1/len(G))**0.5

def _induced_norm_vec_rms_l2(G):
    """sqrt(d) * inf norm (1D)."""
    return _norm_inf(G) * (len(G))**0.5


# -----------------------------------------------------------------------------
# Induced norms for 2D arrays
# -----------------------------------------------------------------------------

def _induced_norm_l2_l2(G):
    """Operator norm (2D)."""
    return _norm_op(G)

def _induced_norm_rms_rms(G):
    """sqrt(d_in/d_out) * operator norm (2D)."""
    return _norm_op(G) * (G.shape[1] / G.shape[0])**0.5

def _induced_norm_l2_rms(G):
    """sqrt(1/d_out) * operator norm (2D)."""
    return _norm_op(G) * (1/G.shape[0])**0.5

def _induced_norm_rms_l2(G):
    """sqrt(d_in) * operator norm (2D)."""
    return _norm_op(G) * (G.shape[1])**0.5

def _induced_norm_l1_l2(G):
    """Column max L2 norm (2D)."""
    return _norm_colmax_l2(G)

def _induced_norm_l1_rms(G):
    """sqrt(1/d_out) * column max L2 norm (2D)."""
    return _norm_colmax_l2(G) * (1/G.shape[0])**0.5

def _induced_norm_l1_inf(G):
    """Max norm (2D)."""
    return _norm_max(G)

def _induced_norm_l2_inf(G):
    """Row max L2 norm (2D)."""
    return _norm_rowmax_l2(G)

def _induced_norm_rms_inf(G):
    """sqrt(d_in) * row max L2 norm (2D)."""
    return _norm_rowmax_l2(G) * (G.shape[1])**0.5


# -----------------------------------------------------------------------------
# Main tensor norm function factory
# -----------------------------------------------------------------------------

def get_tensor_norm_fn(norm: str) -> ArrayFn:
    """Wraps all tensor norm functions."""
    norm_map = {
        # 1d norms
        "l2": _norm_l2,
        "l1": _norm_l1,
        "inf_": _norm_inf,
        "rms": _norm_rms,
        # 2d norms
        "op": _norm_op,
        "fro": _norm_fro,
        "max": _norm_max,
        "colmax_l2": _norm_colmax_l2,
        "rowmax_l2": _norm_rowmax_l2,
        # induced 1d norms
        "vec_l2_l2": _induced_norm_vec_l2_l2,
        "vec_l2_rms": _induced_norm_vec_l2_rms,
        "vec_rms_l2": _induced_norm_vec_rms_l2,
        # induced 2d norms
        "l2_l2": _induced_norm_l2_l2,
        "rms_rms": _induced_norm_rms_rms,
        "l2_rms": _induced_norm_l2_rms,
        "rms_l2": _induced_norm_rms_l2,
        "l1_l2": _induced_norm_l1_l2,
        "l1_rms": _induced_norm_l1_rms,
        "l1_inf": _induced_norm_l1_inf,
        "l2_inf": _induced_norm_l2_inf,
        "rms_inf": _induced_norm_rms_inf,
    }
    if norm not in norm_map:
        raise ValueError(f"Invalid norm type: '{norm}'.")
    tensor_norm_fn = norm_map[norm]
    return tensor_norm_fn
