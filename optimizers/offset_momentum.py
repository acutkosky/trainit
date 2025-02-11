import jax
from jax import  numpy as jnp
import optax
from typing import NamedTuple, Optional


class OffsetMomentumState(NamedTuple):
    offset: optax.Params


def offset_momentum(beta:float= 0.99) -> optax.GradientTransformation:

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


class ImplicitTransportState(NamedTuple):
    previous_updates: Optional[optax.Updates]
    count: int
    


def implicit_transport(beta: float) -> optax.GradientTransformation:
    def init_fn(params: optax.Params):
        return ImplicitTransportState(
            previous_updates = jax.tree.map(jnp.zeros_like, params),
            count=0,
        )

    def update_fn(updates: optax.Updates, state: ImplicitTransportState, params: Optional[optax.Params]=None):

        next_updates = jax.tree.map(
            lambda u, p_u: u + ((state.count!=0)*beta/(1.0-beta)) * (u - p_u),
            updates,
            state.previous_updates
        )

        next_state = ImplicitTransportState(
            previous_updates=updates,
            count=state.count+1,
        )

        return next_updates, next_state

    return optax.GradientTransformation(init_fn, update_fn)
