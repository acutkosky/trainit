import optax
import jax
from jax import numpy as jnp
from jax import tree_util as jtu
from optax import tree_utils as otu
from typing import NamedTuple, Optional
from logstate import Log


class VolumizeState(NamedTuple):
    norms: optax.Params
    logs: Log


# apply volumization, inspired by https://arxiv.org/abs/2003.11243
# basically, for any parameter for which the weights are a matrix
# we will ensure that the final update does  ont change the norm
# of the weights.
# Alternatively, instead of applying this constriant to matrix parameters
# one can provide a boolean pytree that specified which params
# to apply it to.
def volumize(
    params_to_volumize: Optional[optax.Params] = None,
    eps: float = 1e-8,
    stay_on_sphere=True,
):
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
        return VolumizeState(
            norms=norms,
            logs=Log(
                {
                    "volumization/norm_before_volumization": jnp.array(0.0),
                    "volumization/norm_after_volumization": jnp.array(0.0),
                    "volumization/norm_before_minus_norm_after": jnp.array(0.0),
                    "volumization/matrices_norm_before_volumization": jnp.array(0.0),
                    "volumization/matrices_norm_after_volumization": jnp.array(0.0),
                    "volumization/matrices_norm_before_minus_norm_after": jnp.array(
                        0.0
                    ),
                }
            ),
        )

    def update_fn(updates: optax.Updates, state: VolumizeState, params: optax.Params):
        unprojected_next_params = optax.apply_updates(params, updates)

        def project_params(param, norm):
            if norm is None:
                return param
            if stay_on_sphere:
                return param / (jnp.linalg.norm(param) + eps) * norm
            else:
                return param * jnp.minimum(1.0, norm / (jnp.linalg.norm(param) + eps))

        next_params = jtu.tree_map(project_params, unprojected_next_params, state.norms)
        pre_norm = otu.tree_l2_norm(unprojected_next_params)
        post_norm = otu.tree_l2_norm(next_params)

        def matrix_only_norms(x):
            if x.ndim != 2:
                return 0.0
            else:
                return jnp.linalg.norm(x) ** 2

        pre_norm_matrix = jnp.sqrt(
            otu.tree_sum(jtu.tree_map(matrix_only_norms, unprojected_next_params))
        )
        post_norm_matrix = jnp.sqrt(
            otu.tree_sum(jtu.tree_map(matrix_only_norms, next_params))
        )

        next_updates = otu.tree_sub(next_params, params)

        state = VolumizeState(
            norms=state.norms,
            logs=Log(
                {
                    "volumization/norm_before_volumization": pre_norm,
                    "volumization/norm_after_volumization": post_norm,
                    "volumization/norm_before_minus_norm_after": pre_norm - post_norm,
                    "volumization/matrices_norm_before_volumization": pre_norm_matrix,
                    "volumization/matrices_norm_after_volumization": post_norm_matrix,
                    "volumization/matrices_norm_before_minus_norm_after": pre_norm_matrix
                    - post_norm_matrix,
                }
            ),
        )

        return next_updates, state

    return optax.GradientTransformation(init_fn, update_fn)


class CorrectedVolumizationState(NamedTuple):
    base_state: optax.OptState
    volumize_state: optax.OptState
    volumized_direction: optax.Updates
    logs: Log


def corrected_volumization(
    base_tx: optax.GradientTransformation, always_correct=False, **volumization_args
):
    volumize_tx = volumize(**volumization_args)

    def init_fn(params):
        base_state = base_tx.init(params)
        volumize_state = volumize_tx.init(params)
        volumized_direction = jtu.tree_map(lambda p: 0.0, params)

        return CorrectedVolumizationState(
            base_state=base_state,
            volumize_state=volumize_state,
            volumized_direction=volumized_direction,
            logs=Log(
                {
                    "volumization/corrected/original_grad_norm": jnp.array(0.0),
                    "volumization/corrected/corrected_grad_norm": jnp.array(0.0),
                    "volumization/corrected/original_minus_corrected_grad_norm": jnp.array(
                        0.0
                    ),
                    "volumization/corrected/cos(grad,matrix_params)": jnp.array(0.0),
                    "volumization/corrected/cos(corrected_grad,matrix_params)": jnp.array(
                        0.0
                    ),
                }
            ),
        )

    def update_fn(updates, state, params):
        orig_grad_norm = otu.tree_l2_norm(updates)

        def project_out_param(u_i, p_i, direction_i):
            p_norm = jnp.linalg.norm(p_i) + 1e-8
            vdot = jnp.dot(p_i.flatten() / p_norm, u_i.flatten())
            if not always_correct:
                vdot = direction_i * jax.nn.relu(direction_i * vdot)
            component = vdot * p_i / (jnp.linalg.norm(p_i) + 1e-8)
            return u_i - component

        corrected_updates = jtu.tree_map(
            project_out_param, updates, params, state.volumized_direction
        )

        corrected_grad_norm = otu.tree_l2_norm(corrected_updates)

        base_updates, next_base_state = base_tx.update(
            corrected_updates, state.base_state, params
        )

        volumized_updates, next_volumize_state = volumize_tx.update(
            base_updates, state.volumize_state, params
        )

        # direction is 1 if volumization increased the norm, -1 if it decreased the norm,
        # 0 otherwise.
        next_direction = jtu.tree_map(
            lambda p, vu, bu: jnp.sign(
                jnp.linalg.norm(p + vu) - jnp.linalg.norm(p + bu)
            ),
            params,
            volumized_updates,
            base_updates,
        )

        def dot_prod_matrix_only(param, grad):
            if param.ndim != 2:
                return 0.0
            else:
                return jnp.dot(param.flatten(), grad.flatten())

        cos_sim = jtu.tree_map(dot_prod_matrix_only, params, updates)
        cos_sim = otu.tree_sum(cos_sim)
        cos_sim = cos_sim / (
            otu.tree_l2_norm(updates) * otu.tree_l2_norm(params) + 1e-8
        )

        corrected_cos_sim = jtu.tree_map(
            dot_prod_matrix_only, params, corrected_updates
        )
        corrected_cos_sim = otu.tree_sum(corrected_cos_sim)
        corrected_cos_sim = corrected_cos_sim / (
            otu.tree_l2_norm(corrected_updates) * otu.tree_l2_norm(params) + 1e-8
        )
        logs = Log(
            {
                "volumization/corrected/original_grad_norm": orig_grad_norm,
                "volumization/corrected/corrected_grad_norm": corrected_grad_norm,
                "volumization/corrected/original_minus_corrected_grad_norm": orig_grad_norm
                - corrected_grad_norm,
                "volumization/corrected/cos(grad,matrix_params)": cos_sim,
                "volumization/corrected/cos(corrected_grad,matrix_params)": corrected_cos_sim,
            }
        )
        next_state = CorrectedVolumizationState(
            base_state=next_base_state,
            volumize_state=next_volumize_state,
            volumized_direction=next_direction,
            logs=logs,
        )
        return volumized_updates, next_state

    return optax.GradientTransformation(init_fn, update_fn)
