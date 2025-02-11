"""Optimizer algorithms."""

import jax
import jax.tree_util as jtu
import jax.random as jr
import optax
import optimizers
import equinox as eqx
import copy
from omegaconf import DictConfig, OmegaConf
from typing import Any, Tuple
from jaxtyping import PRNGKeyArray


import sys
sys.path.append('/projectnb/aclab/cutkosky/jaxol')
import jaxol.default as precond
import jaxol.prec_adam as prec_adam

def init_schedule(lr_config: DictConfig) -> optax.ScalarOrSchedule:
    """Parses the config and initializes a learning rate scheduler.

    Args:
        lr_config: The learning rate config.
        kargs: Additional arguments to overwrite learning rate config.

    Returns:
        A `optax.ScalarOrSchedule` object.
    """
    is_positive_int = lambda x: isinstance(x, int) and (x > 0)

    def init_constant_lr(config):
        learning_rate = config.lr
        return learning_rate
    
    def init_cosine_lr(config):
        if is_positive_int(config.warmup):
            learning_rate = optax.warmup_cosine_decay_schedule(
                init_value=0.0,
                peak_value=config.lr,
                warmup_steps=config.warmup,
                decay_steps=config.max_steps,
            )
        else:
            learning_rate = optax.cosine_decay_schedule(
                init_value=config.lr,
                decay_steps=config.max_steps,
            )
        return learning_rate
    
    def init_linear_lr(config):
        warmup_steps = config.warmup if is_positive_int(config.warmup) else 0
        const_steps = config.const if is_positive_int(config.const) else 0
        learning_rate = optimizers.warmup_const_linear_decay_schedule(
            peak_value=config.lr,
            warmup_steps=warmup_steps,
            const_steps=const_steps,
            total_steps=config.max_steps,
            init_value=0.0,
            end_value=config.end_value,
        )
        return learning_rate
    
    def init_trapezoid_lr(config):
        warmup_steps = config.warmup if is_positive_int(config.warmup) else 0
        decay_steps = config.decay if is_positive_int(config.decay) else 0
        learning_rate = optimizers.trapezoid_schedule(
            peak_value=config.lr,
            total_steps=config.max_steps,
            warmup_steps=warmup_steps,
            decay_steps=decay_steps,
        )
        return learning_rate

    def init_piecewise_linear_lr(config):
        learning_rate = optax.linear_schedule(
            init_value=config.lr1,
            end_value=config.lr2,
            transition_steps=config.max_steps,
            transition_begin=config.start_steps,    # NOTE: for now, we still need to specify the start iteration in config.
        )
        return learning_rate

    if lr_config.schedule == "constant":
        learning_rate = init_constant_lr(lr_config)
    elif lr_config.schedule == "cosine":
        learning_rate = init_cosine_lr(lr_config)
    elif lr_config.schedule == "linear":
        learning_rate = init_linear_lr(lr_config)
    elif lr_config.schedule == "trapezoid":
        learning_rate = init_trapezoid_lr(lr_config)
    elif lr_config.schedule == "piecewise_linear":
        learning_rate = init_piecewise_linear_lr(lr_config)
    else:
        raise ValueError(f"schedule type {lr_config.schedule} is not supported.")
    return learning_rate


def wrap_scheduler(
    learning_rate: optax.ScalarOrSchedule,
    wandb_log: None,
    schedule_title: str="schedule",
):
    """Returns a wrapped scheduler that logs current learning rate.
    
    The wrapped schedule takes in `learning_rate` as argument and returns a scalar lr.
    """
    def wrapper(schedule, count, wandb_log, title):
        if callable(schedule):
            lr = schedule(count)
        else:
            lr = schedule
        if wandb_log is not None:
            jax.experimental.io_callback(wandb_log, None, {f"lr/{title}": lr}, commit=False)
        return lr
    return jtu.Partial(wrapper, learning_rate, wandb_log=wandb_log, title=schedule_title)

def init_optimizer(
    model: eqx.Module,
    config: DictConfig,
    wandb_log: None,
    *,
    key: PRNGKeyArray,
) -> Tuple[optax.GradientTransformation, optax.OptState]:
    """Construct optimizer from model and training config.

    Args:
        model: an equinox.Module object.
        config: global_config.
        wandb_log: optional logger to handle backend wandb logging while training.
        key: random key for optimizer.

    Returns:
        A tuple of optax.GradientTransofrmation and optax.OptState.
    """
    def init_adam_base(config: DictConfig):
        learning_rate = wrap_scheduler(
            init_schedule(config.lr_config), wandb_log=wandb_log)
        return optimizers.adam_base(
            learning_rate=learning_rate,
            beta1=config.beta1,
            beta2=config.beta2,
            eps=config.eps,
            weight_decay=config.weight_decay,
            use_nesterov=config.use_nesterov,
            debias_beta1=config.debias_beta1,
            debias_beta2=config.debias_beta2,
            use_momentum=config.use_momentum,
            use_momentum_state=config.use_momentum_state,
            use_precond=config.use_precond,
            use_precond_state=config.use_precond_state,
            logger=None,
        )
    
    def init_adam(config: DictConfig):
        learning_rate = wrap_scheduler(
            init_schedule(config.lr_config), wandb_log=wandb_log)
        return optimizers.adam(
            learning_rate=learning_rate,
            beta1=config.beta1,
            beta2=config.beta2,
            eps=config.eps,
            weight_decay=config.weight_decay,
        )

    def init_adamw(config: DictConfig):
        learning_rate = wrap_scheduler(
            init_schedule(config.lr_config), wandb_log=wandb_log)
        return optimizers.adamw(
            learning_rate=learning_rate,
            beta1=config.beta1,
            beta2=config.beta2,
            eps=config.eps,
            inner_eps=config.inner_eps,
            weight_decay=config.weight_decay,
            use_nesterov=config.use_nesterov,
        )

    def init_nadam(config: DictConfig):
        learning_rate = wrap_scheduler(
            init_schedule(config.lr_config), wandb_log=wandb_log)
        return optimizers.nadam(
            learning_rate=learning_rate,
            beta1=config.beta1,
            beta2=config.beta2,
            eps=config.eps,
            weight_decay=config.weight_decay,
            decouple_weight_decay=config.decouple_weight_decay,
        )

    def init_rmsprop(config: DictConfig):
        learning_rate = wrap_scheduler(
            init_schedule(config.lr_config), wandb_log=wandb_log)
        return optimizers.rmsprop(
            learning_rate=learning_rate,
            momentum=config.momentum,
            eps=config.eps,
            weight_decay=config.weight_decay,
            decouple_weight_decay=config.decouple_weight_decay,
            inner_eps=config.inner_eps,
        )

    def init_sgdm(config: DictConfig):
        learning_rate = wrap_scheduler(
            init_schedule(config.lr_config), wandb_log=wandb_log)
        return optimizers.sgdm(
            learning_rate=learning_rate,
            momentum=config.momentum,
            use_nesterov=config.use_nesterov,
            weight_decay=config.weight_decay,
            decouple_weight_decay=config.decouple_weight_decay,
        )
    
    def init_muon(config: DictConfig):
        muon_lr = wrap_scheduler(
            init_schedule(config.lr_config), wandb_log=wandb_log)
        adam_lr_config = copy.deepcopy(config.lr_config)
        adam_lr_config.lr = config.adam_lr
        adam_lr = wrap_scheduler(
            init_schedule(adam_lr_config), wandb_log=wandb_log, schedule_title="adam_schedule")
        return optimizers.muon(
            learning_rate=muon_lr,
            momentum=config.momentum,
            nesterov=config.nesterov,
            ns_steps=config.ns_steps,
            use_l2=config.use_l2,
            adam_lr=adam_lr,
            adam_beta1=config.adam_beta1,
            adam_beta2=config.adam_beta2,
            adam_eps=config.adam_eps,
            adam_wd=config.adam_wd,
            offset_beta=config.muon_offset_beta,
            beta2=config.muon_beta2,
        )

    def init_prec_adam(config: DictConfig):
        learning_rate = wrap_scheduler(
            init_schedule(config.lr_config), wandb_log=wandb_log)
        kwargs = OmegaConf.to_container(config)
        del kwargs['lr_config']
        del kwargs['name']
        del kwargs['weight_decay']
        return prec_adam.adam_prec(
            learning_rate=learning_rate,
            **kwargs
        )
        #     b1=config.b1,
        #     b2=config.b2,
        #     b3=config.b3,
        #     pre_diag=config.pre_diag,
        #     post_diag=config.post_diag,
        #     pre_matrix_diag=config.pre_matrix_diag,
        #     do_matrix=config.do_matrix,
        #     eps=config.eps,
        #     solver=config.solver,
        #     eps_type=config.eps_type,
        #     inner_eps=config.inner_eps,
        #     outer_eps=config.outer_eps,
        #     threshold=config.threshold,
        #     offset_beta=config.offset_beta,
        #     wd=config.wd,
        # )


    # Initialize base optimizer.
    name = config.optimizer.name
    opt_config = config.optimizer
    if name == "adam_base":
        optimizer = init_adam_base(opt_config)
    elif name == "adam":
        optimizer = init_adam(opt_config)
    elif name == "adamw":
        optimizer = init_adamw(opt_config)
    elif name == "nadam":
        optimizer = init_nadam(opt_config)
    elif name == "rmsprop":
        optimizer = init_rmsprop(opt_config)
    elif name == "sgdm":
        optimizer = init_sgdm(opt_config)
    elif name == "muon":
        optimizer = init_muon(opt_config)
    elif name == "precond":
        optimizer = precond.get_optimizer(conf=opt_config)
    elif name == 'prec_adam':
        optimizer = init_prec_adam(config=opt_config)

    if opt_config.offset_beta:
        optimizer = optax.chain(optimizer, optimizers.offset_momentum(opt_config.offset_beta))
    if opt_config.igt_beta:
        optimizer = optax.chain(optimizer, optimizers.implicit_transport(opt_config.igt_beta))
    # Wrap online-to-nonconvex.
    if name in ["ogd_md"]:
        wrap_o2nc = True
    elif name in ["adamw", "sgdm", "polar", "jump"]:
        wrap_o2nc = False
    else:
        wrap_o2nc = config.train.wrap_o2nc
    if wrap_o2nc:
        optimizer = optimizers.online_to_gradient_transformation(optimizer)

    # Wrap random scaling.
    random_scaling_key, key = jr.split(key)
    optimizer = optimizers.wrap_random_scaling(
        gradient_transformation=optimizer,
        random_scaling=config.train.random_scaling,
        use_importance_sampling=config.train.use_importance_sampling,
        key=random_scaling_key,
    )
    
    # Gradient clipping and finite gradient wrapper.
    grad_clip = optax.clip_by_global_norm(config.train.gradient_clip_val)
    optimizer = optax.chain(
        grad_clip,
        optax.apply_if_finite(optimizer, 15)
    )

    # Initialize opt_state
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    return optimizer, opt_state
