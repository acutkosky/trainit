import optax
import jax
from jax import numpy as jnp
from jax import tree_util as jtu
from optax import tree_utils as otu
from typing import NamedTuple, Optional


class VolumizeState(NamedTuple):
    norms: optax.Params


# apply volumization, inspired by https://arxiv.org/abs/2003.11243
# basically, for any parameter for which the weights are a matrix
# we will ensure that the final update does  ont change the norm
# of the weights.
# Alternatively, instead of applying this constriant to matrix parameters
# one can provide a boolean pytree that specified which params
# to apply it to.
def volumize(params_to_volumize: Optional[optax.Params] = None, eps: float = 1e-8, stay_on_sphere=True):
    def init_fn(params: optax.Params):
        def is_matrix(p):
            return p.ndim == 2

        def create_norms(p, mask):
            if mask is None or jnp.linalg.norm(p) == 0:
                return None
            return jnp.linalg.norm(p)

        if params_to_volumize is None:
            mask = jtu.tree_map(is_matrix, params)
        else:
            mask = params_to_volumize

        norms = jtu.tree_map(create_norms, params, mask)
        return VolumizeState(norms=norms)

    def update_fn(updates: optax.Updates, state: VolumizeState, params: optax.Params):
        unprojected_next_params = optax.apply_updates(params, updates)

        def project_params(param, norm):
            if norm is None:
                return param
            if stay_on_sphere:
                return param / (jnp.linalg.norm(param) + eps) * norm
            else:
                return param * jnp.minimum(1.0, norm/(jnp.linalg.norm(param) + eps))

        next_params = jtu.tree_map(project_params, unprojected_next_params, state.norms)

        next_updates = otu.tree_sub(next_params, params)

        return next_updates, state

    return optax.GradientTransformation(init_fn, update_fn)
