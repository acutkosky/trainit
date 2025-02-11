#!/bin/bash -l

#$ -pe omp 8
#$ -l gpus=1
#$ -l gpu_c=7.0
#$ -l gpu_memory=40G    # Specifies GPU memory
#$ -l h_rt=12:00:00      # Specifies the hard time limit for the job

cd /projectnb/aclab/cutkosky/qinzitrainit/trainit
module load python3/3.10.12 cuda/12.2
source env/bin/activate
# project=null
project=test_offset
# run=precond_norm_yesavg_gscale_wrp2_cap9999_binv_perlayer
steps=4000
batch_size=128

wd=0.1
nesterov=True
use_momentum=True
use_preconditioning=True
decouple_weight_decay=False
debias_beta1=True
debias_beta2=True

schedule=linear
lr=1e-3
warmup=200
end_value=0.0

# BELOW for saving checkpoint without loggings
# project=null
# log_data=False

log_data=False

save_checkpoint=False
save_path="checkpoint/$project/precond_baseline"
save_steps="[2000]"

optimizer=adam
run=trainit_adam_base_b19_b299_wd0.1
b1=0.9
b2=0.99
wd=0.1
python main.py \
    logging.wandb_project='offset_momentum' \
    logging.wandb_name=$run \
    logging.log_callback_data=$log_data \
    train.max_steps=$steps \
    dataset.total_batch_size=$batch_size \
    optimizer=$optimizer \
    checkpoint.save=$save_checkpoint \
    checkpoint.save_path=$save_path \
    checkpoint.save_steps=$save_steps \
    optimizer.weight_decay=$wd \
    optimizer.beta1=$b1 \
    optimizer.beta2=$b2 \
    optimizer.inner_eps=0.0 \
    optimizer.eps=1e-8 \
    optimizer/lr_config=$schedule \
    optimizer.lr_config.lr=$lr \
    optimizer.lr_config.warmup=$warmup \
    optimizer.lr_config.max_steps=$steps \
    optimizer.lr_config.end_value=$end_value

# run=precond_mat_norm_yavg_gsc_wrp3_cap1.0_binv_1posreg_ynorm_t4096_eps1.0
# run=precond_out_prediag_norm_yesavg_gsc_wrp2_cap1.0_avgconv_1posreg_ynorm_t4096_eps1.0
# optimizer=precond
# python main.py \
#     logging.wandb_project='scratch' \
#     logging.wandb_name=$run \
#     logging.log_callback_data=$log_data \
#     train.max_steps=$steps \
#     dataset.total_batch_size=$batch_size \
#     optimizer=$optimizer \
#     checkpoint.save=$save_checkpoint \
#     checkpoint.save_path=$save_path \
#     checkpoint.save_steps=$save_steps \
#     optimizer.do_average_offset=True \
#     optimizer.direction='precond' \
#     +optimizer.direction_kwargs.eps_type='norm' \
#     +optimizer.direction_kwargs.prec_type='outer' \
#     +optimizer.direction_kwargs.pre_diag=0.25 \
#     +optimizer.direction_kwargs.threshold=4096 \
#     +optimizer.direction_kwargs.eps=1.0 \
#     +optimizer.direction_kwargs.normalize=True \
#     +optimizer.direction_kwargs.solver='conjgrad_5' \
#     +optimizer.direction_kwargs.inner_eps=1e-8 \
#     +optimizer.direction_kwargs.outer_eps=0.0 \
#     +optimizer.direction_kwargs.pre_matrix_diag=0.5 \
#     optimizer.offset.filter=False \
#     optimizer.offset.grad_scale=False \
#     optimizer.averaging.weight_ratio='poly_cap' \
#     optimizer.averaging.wr.power=2.0 \
#     optimizer.averaging.beta='invweight' \
#     optimizer.averaging.wr.cap=0.9 \
#     optimizer.averaging.weight_decay=0.0 \
#     optimizer.use_unit='per_layer' \
#     optimizer.offset.grad_scale=True \
#     +optimizer.scale_kwargs.eps=1e-4 \
#     +optimizer.scale_kwargs.k=4 \
#     +optimizer.scale_kwargs.minvalue=0.0 \
#     +optimizer.scale_kwargs.pos_reg=1.0 \
#     optimizer.add_conv.scale_type='const' \
#     optimizer.add_conv.key=31 \
#     optimizer.conversion='average'

# run=diag_eps1-e8_yesnorm_yesavg_gscale_wrp3_cap0.99_binv_wd0.0_1posreg
# optimizer=precond
# python main.py \
#     logging.wandb_project=$project \
#     logging.wandb_name=$run \
#     logging.log_callback_data=$log_data \
#     train.max_steps=$steps \
#     dataset.total_batch_size=$batch_size \
#     optimizer=$optimizer \
#     checkpoint.save=$save_checkpoint \
#     checkpoint.save_path=$save_path \
#     checkpoint.save_steps=$save_steps \
#     optimizer.direction='diag' \
#     +optimizer.direction_kwargs.eps=1e-8 \
#     +optimizer.direction_kwargs.normalize=True \
#     optimizer.offset.filter=False \
#     optimizer.do_average_offset=True \
#     optimizer.averaging.wr.power=3.0 \
#     optimizer.averaging.beta='invweight' \
#     optimizer.averaging.wr.cap=0.99 \
#     optimizer.averaging.weight_decay=0.0 \
#     optimizer.use_unit='per_layer' \
#     optimizer.offset.grad_scale=True \
#     optimizer.conversion='average' \
#     optimizer.scale='pos_log_cb' \
#     +optimizer.scale_kwargs.eps=1e-4 \
#     +optimizer.scale_kwargs.k=4 \
#     +optimizer.scale_kwargs.minvalue=0.0 \
#     +optimizer.scale_kwargs.pos_reg=1.0 \

# optimizer=prec_adam
# run=offset_momentum99_b99_wd0
# # run=base_adamw_wd1
# run=redo_offset_momentum99_wd1
# # run=base_adamw_wd0
# # run=offset_momentum99_b999_wd1
# # run=base_adamw_b99_wd0
# # run=offset_momentum99_b99_wd0_post25_pre25
# python main.py \
#     logging.wandb_project='offset_momentum' \
#     logging.wandb_name=$run \
#     logging.log_callback_data=$log_data \
#     train.max_steps=$steps \
#     dataset.total_batch_size=$batch_size \
#     optimizer=$optimizer \
#     checkpoint.save=$save_checkpoint \
#     checkpoint.save_path=$save_path \
#     checkpoint.save_steps=$save_steps \
#     optimizer.weight_decay=$wd \
#     optimizer.use_nesterov=$nesterov \
#     optimizer.b1=0.9 \
#     optimizer.b2=0.99 \
#     optimizer.b3=0.99 \
#     optimizer.pre_diag=0.5 \
#     optimizer.post_diag=0.0 \
#     optimizer.pre_matrix_diag=0.5 \
#     optimizer.do_matrix=False \
#     optimizer.offset_beta=0.99 \
#     optimizer.wd=0.1 \
#     optimizer.eps=1.0 \
#     optimizer.inner_eps=0.0 \
#     optimizer.outer_eps=1e-8 \
#     optimizer.solver='conjgradfix_5' \
#     optimizer.eps_type='trace' \
#     optimizer.threshold=4096 \
#     optimizer/lr_config=$schedule \
#     optimizer.lr_config.lr=$lr \
#     optimizer.lr_config.warmup=$warmup \
#     optimizer.lr_config.max_steps=$steps \
#     optimizer.lr_config.end_value=$end_value

# optimizer=precond
# run=md_lr1e-4_beta0.999_noavg_gscale_wr0.9_b1_wd0.0
# python main.py \
#     logging.wandb_project=$project \
#     logging.wandb_name=$run \
#     logging.log_callback_data=$log_data \
#     train.max_steps=$steps \
#     dataset.total_batch_size=$batch_size \
#     optimizer=$optimizer \
#     checkpoint.save=$save_checkpoint \
#     checkpoint.save_path=$save_path \
#     checkpoint.save_steps=$save_steps \
#     optimizer.do_average_offset=False \
#     optimizer.direction_kwargs.eps_type='norm' \
#     optimizer.offset.filter=False \
#     optimizer.averaging.weight_ratio=0.9 \
#     optimizer.averaging.beta=1.0 \
#     optimizer.averaging.weight_decay=0.0 \
#     optimizer.use_unit=False \
#     optimizer.offset.grad_scale=True \
#     optimizer.scale='mirror_descent' \
#     +optimizer.scale_kwargs.beta=0.999 \
#     +optimizer.scale_kwargs.lr_rescale=1e-4


# run=ftrl_scale_add_lr1e-3_beta2null_wrc_yrescale_beta10.9
# optimizer=precond
# python main.py \
#     logging.wandb_project=$project \
#     logging.wandb_name=$run \
#     logging.log_callback_data=$log_data \
#     train.max_steps=$steps \
#     dataset.total_batch_size=$batch_size \
#     optimizer=$optimizer \
#     checkpoint.save=$save_checkpoint \
#     checkpoint.save_path=$save_path \
#     checkpoint.save_steps=$save_steps \
#     optimizer.do_average_offset=False \
#     optimizer.direction='precond' \
#     +optimizer.direction_kwargs.eps_type='norm' \
#     +optimizer.direction_kwargs.prec_type='outer' \
#     +optimizer.direction_kwargs.threshold=4096 \
#     +optimizer.direction_kwargs.eps=1.0 \
#     +optimizer.direction_kwargs.normalize=True \
#     +optimizer.direction_kwargs.solver='solve_inc' \
#     optimizer.offset.filter=False \
#     optimizer.offset.grad_scale=True \
#     optimizer.averaging.weight_ratio=0.9 \
#     optimizer.averaging.wr.power=1.0 \
#     optimizer.averaging.beta='invweight' \
#     optimizer.averaging.wr.cap=0.9 \
#     optimizer.averaging.weight_decay=0.0 \
#     optimizer.use_unit=False \
#     optimizer.offset.grad_scale=True \
#     optimizer.add_conv.scale_type='const' \
#     optimizer.add_conv.key=31 \
#     optimizer.scale='ftrl' \
#     +optimizer.scale_kwargs.lr=1e-3 \
#     +optimizer.scale_kwargs.denom_beta=null \
#     +optimizer.scale_kwargs.lr_rescale=True \
#     optimizer.conversion='add' \
#     optimizer.add_conv.scale_type='const' \
#     optimizer.add_conv.key=31


# optimizer=precond
# run=ftrl_scale_add_lr1e-3_beta20.999_beta10.9_2000
# python main.py \
#     logging.wandb_project=$project \
#     logging.wandb_name=$run \
#     logging.log_callback_data=$log_data \
#     train.max_steps=$steps \
#     dataset.total_batch_size=$batch_size \
#     optimizer=$optimizer \
#     checkpoint.save=$save_checkpoint \
#     checkpoint.save_path=$save_path \
#     checkpoint.save_steps=$save_steps \
#     optimizer.do_average_offset=False \
#     optimizer.direction_kwargs.eps_type='norm' \
#     optimizer.offset.filter=False \
#     optimizer.averaging.weight_ratio=0.9 \
#     optimizer.averaging.beta=1.0 \
#     optimizer.averaging.weight_decay=0.0 \
#     optimizer.use_unit=False \
#     optimizer.offset.grad_scale=True \
#     optimizer.scale='ftrl' \
#     +optimizer.scale_kwargs.lr=1e-3 \
#     +optimizer.scale_kwargs.denom_beta=0.999 \
#     +optimizer.scale_kwargs.lr_rescale=True \
#     optimizer.conversion='add'


# optimizer=adam
# run=trainit_adam_base_b19_b299_wd0.1
# b1=0.9
# b2=0.99
# wd=0.1
# python main.py \
#     logging.wandb_project='offset_momentum' \
#     logging.wandb_name=$run \
#     logging.log_callback_data=$log_data \
#     train.max_steps=$steps \
#     dataset.total_batch_size=$batch_size \
#     optimizer=$optimizer \
#     checkpoint.save=$save_checkpoint \
#     checkpoint.save_path=$save_path \
#     checkpoint.save_steps=$save_steps \
#     optimizer.weight_decay=$wd \
#     optimizer.use_momentum=$use_momentum \
#     optimizer.debias_beta1=$debias_beta1 \
#     optimizer.debias_beta2=$debias_beta2 \
#     optimizer.beta1=$b1 \
#     optimizer.beta2=$b2 \
#     optimizer.inner_eps=0.0 \
#     optimizer.eps=1e-8 \
#     optimizer.use_preconditioning=$use_preconditioning \
#     optimizer.decouple_weight_decay=$decouple_weight_decay \
#     optimizer/lr_config=$schedule \
#     optimizer.lr_config.lr=$lr \
#     optimizer.lr_config.warmup=$warmup \
#     optimizer.lr_config.max_steps=$steps \
#     optimizer.lr_config.end_value=$end_value
