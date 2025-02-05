import jax
from jax import  numpy as jnp
import optax
from typing import NamedTuple, Optional


class OffsetMomentumState:
    offset: optax.Params


def offset_momentum(beta:float= 0.99) -> optax.GradienTransforamtion:

    def init_fn(params: optax.Params):
        return OffsetMomentumState(
            offset=jax.tree.map(jnp.zeros_like, params)
        )

    def update_fn(updates: optax.Updates, state: OffsetMomentumState, params: Optional[optax.Params]=None):

        next_offset = jax.tree.map(
            lambda o, u: o * beta + u * (1-beta),
            state.offset,
            updates
        )

        next_state = OffsetMomentumState(offset=next_offset)

        updates = jax.tree.map(
            lambda u, o: u + o,
            updates,
            next_offset
        )

        return updates, next_state

    return optax.GradientTransformation(init_fn, update_fn)    
