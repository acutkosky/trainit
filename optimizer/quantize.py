import jax.numpy as jnp
from jax import custom_jvp, custom_vjp
import jax


def differentiable_sign(
    x: jax.typing.ArrayLike,
    sample_method: str = "zero",
    clip: bool = False,
    rms_scale: bool = True,
    key: jax.typing.ArrayLike = None,
):
    assert sample_method in [
        "zero",
        "random",
    ], f"invalid sample_method: {sample_method}"
    if sample_method == "random":
        assert (
            key is not None
        ), f"if sample_method='random', you must supply a prng key!"

    @custom_vjp
    def sign_fn(x: jax.typing.ArrayLike, key: jax.typing.ArrayLike = None):
        if sample_method == "zero":
            alpha = jnp.zeros_like(x)
        elif sample_method == "random":
            alpha = jnp.random.uniform(key=key, shape=x.shape, minval=-1, maxval=1)

        if clip:
            pre_sign = jnp.clip(x, -1, 1)
        else:
            pre_sign = x
        result = jnp.sign(pre_sign - alpha)
        return result

    def sign_fwd(x: jax.typing.ArrayLike, key: jax.typing.ArrayLike = None):
        return sign_fn(x, key), x

    def sign_bwd(x, grad_out):
        if clip:
            new_grad = grad_out * (jnp.sign(grad_out) * (x - jnp.clip(x, -1, 1)) >= 0)
        else:
            new_grad = grad_out
        return new_grad, None

    sign_fn.defvjp(sign_fwd, sign_bwd)

    result = sign_fn(x, key)
    if rms_scale:
        result = result * jnp.sqrt(jnp.mean(x**2))
    return result


