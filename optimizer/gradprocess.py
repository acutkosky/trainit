import jax
from jax import numpy as jnp
import optax
from jax import tree_util as jtu
from optax import tree_utils as otu
from typing import NamedTuple, Optional
from logstate import Log


class CorrectProjectionState(NamedTuple):
    correction_vector: optax.Updates
    base_state: optax.OptState
    proj_state: optax.OptState
    # mask: optax.Updates


def tree_remove_signed_projection(base, proj_v, sign=1.0, ord=2):
    assert ord == 2  # we only handle l2 for now

    vdot = otu.tree_vdot(base, proj_v)

    if sign is not None:
        vdot = sign*jax.nn.relu(sign*vdot)

    component = otu.tree_scalar_mul(vdot, proj_v)

    return otu.tree_sub(base, component)

def remove_signed_projection(base, proj_v, filter_fn: lambda b: True, sign=1.0, ord=2):

    if not filter_fn(base):
        return base
        
    if proj_v is None:
        return base
    
    assert ord == 2  # we only handle l2 for now
    
    assert base.shape[0] == proj_v.shape[0], f"mismatch: {base.shape}, {proj_v.shape}"
    vdot = jnp.dot(base.flatten(), proj_v.flatten())

    if sign is not None:
        vdot = sign*jax.nn.relu(sign*vdot)

    # compute positive component of base along ray indicated by proj_v * sign
    # If sign is None, then we compute the component along the linear space
    # spanned by proj_v
    component = vdot * proj_v / (jnp.linalg.norm(proj_v, ord=ord)**2 + 1e-8)

    return base  - component


def normalize(v, ord=2, eps=1e-8):
    return v / (jnp.linalg.norm(v, ord=ord) + eps)


def normalize_tree(t, ord=2, eps=1e-8):
    norm = otu.tree_l2_norm(t)

    return otu.tree_scalar_mul(1.0 / (norm + eps), t)


def correct_projections(
    base_tx: optax.GradientTransformation,
    proj_tx: optax.GradientTransformation,
    per_layer: bool = True,
    spherical:bool = True,
    per_layer_matrices_only: bool = True,
    signed: bool = True,
    ord=2,
) -> optax.GradientTransformation:
    """
    applies a correction to the gradient supplied to pre_tx
    to correct for changes applied by post_tx.
    """
    if per_layer_matrices_only:
        filter_fn = lambda b: b.ndim == 2
    else:
        filter_fn = lambda b: True

    def init_fn(params: optax.Params):
        base_state = base_tx.init(params)
        proj_state = proj_tx.init(params)

        if spherical:
            correction_vector = None
        else:
            correction_vector = otu.tree_zeros_like(params)

        # if per_layer_matrices_only:
        #     mask = jtu.tree_map(lambda p: p.ndim == 2, params)
        # else:
        #     mask = jtu.tree_map(lambda p: True, params)

        return CorrectProjectionState(
            correction_vector=correction_vector,
            base_state=base_state,
            proj_state=proj_state,
            # mask=mask
        )

    def update_fn(
        grads: optax.Updates,
        state: CorrectProjectionState,
        params: Optional[optax.Params] = None,
    ):
        if spherical:
            if per_layer:
                # if per_layer_matrices_only:
                #     correction_vector = jtu.tree_map(
                #         lambda p: p if p.ndim == 2 else None,
                #         params
                #     )
                # else:
                correction_vector = params
                # correction_vector = jtu.tree_map(normalize, params)
            else:
                correction_vector = normalize_tree(params, ord)
            sign = -1.0
        else:
            correction_vector = state.correction_vector
            sign = 1.0

        if not signed:
            sign = None

        fixed_grads = grads
        if per_layer:
            fixed_grads = jtu.tree_map(
                lambda g, c: remove_signed_projection(g, c, sign=sign, filter_fn=filter_fn), grads, correction_vector
            )
        else:
            fixed_grad = tree_remove_signed_projection(grads, correction_vector, sign=sign)

        base_updates, next_base_state = base_tx.update(
            fixed_grads, state.base_state, params
        )

        proj_updates, next_proj_state = proj_tx.update(
            base_updates, state.proj_state, params
        )


        if spherical:
            next_correction_vector = None
        else:
            update_diff = otu.tree_sub(proj_updates, base_updates)
            if per_layer:
                next_correction_vector = jtu.tree_map(normalize, update_diff)
            else:
                next_correction_vector = normalize_tree(update_diff, ord)

        next_state = CorrectProjectionState(
            correction_vector=next_correction_vector,
            base_state=next_base_state,
            proj_state=next_proj_state,
            # mask=state.mask
        )

        return proj_updates, next_state

    return optax.GradientTransformation(init_fn, update_fn)


class RandomScaleState(NamedTuple):
    base_state: optax.OptState
    rng_key: jax.Array
    prev_key: jax.Array


def random_scale(
    base_tx: optax.GradientTransformation,
    rng_key: jax.Array,
    distribution: str = "exponential",
):
    assert distribution in ["exponential", "uniform", "per_coord_exponential"]

    def init_fn(params):

        to_use, saved_rng_key = jax.random.split(rng_key)
        
        return RandomScaleState(
            base_tx.init(params),
            saved_rng_key,
            to_use,
            )

    def update_fn(grads, state, params=None):
        to_use, rng_key = jax.random.split(state.rng_key)

        if distribution == "exponential":
            u_scale = jax.random.exponential(to_use)
            g_scale = 1.0

        if distribution == "per_coord_exponential":
            u_scale = otu.tree_random_like(to_use, grads, jax.random.exponential)
            g_scale = 1.0

        elif distribution == "uniform":
            u_scale = jax.random.uniform(to_use, minval=0.0, maxval=2.0)
            prev_scale = jax.random.uniform(state.prev_key, minval=0.0, maxval=2.0)
            g_scale = 2.0 - prev_scale

        scaled_grads = otu.tree_scalar_mul(g_scale, grads)
        updates, next_base_state = base_tx.update(
            scaled_grads, state.base_state, params
        )

        if "per_coord" not in distribution:
            scaled_updates = otu.tree_scalar_mul(u_scale, updates)
        else:
            scaled_updates =  otu.tree_mul(u_scale, updates)



        next_state = RandomScaleState(
            base_state=next_base_state,
            rng_key=rng_key,
            prev_key=to_use
        )

        return scaled_updates, next_state

    return optax.GradientTransformation(init_fn, update_fn)
