# Mango_v3 experiments.

#!/bin/bash

# project=null
project=pile_baseline

log_data=False
steps=2000
batch_size=128
# default to true, only change to false for test purpose
use_amp=True

schedule=linear
warmup=200
const=null

optimizer=mango_v3

exp_name=mango_v3

# System variables
BASE_DIR=/projectnb/aclab/qinziz/trainit
DATE=$(date +"%Y-%m-%d")
OUTPUT_PATH=$BASE_DIR/scc_outputs/$DATE/$exp_name

mkdir -p $OUTPUT_PATH


# ========================================================================
# Mango optimzier configs
# ========================================================================

# ...
# name="mango_v3_baseline"


# ... Head
# ......
# head_normalize="inf_"
# name="mango_v3_head_inf"

# ......
# head_normalize="l2_row"
# head_scale_dim=True
# head_scale_dim_clip_min=null
# name="mango_v3_head_l2-row"
# # <next>
# head_scale_dim=False
# name="mango_v3_head_l2-row_nodim"

# ......
# head_normalize="l2_col"
# head_scale_dim=True
# head_scale_dim_clip_min=null
# name="mango_v3_head_l2-col"
# # <next>
# head_scale_dim=False
# name="mango_v3_head_l2-col_nodim"

# ......
# head_normalize=null
# name="mango_v3_head_null"

# ......
# head_lr=5e-5
# name="mango_v3_head_rms-to-l2_head-lr${head_lr}"
# head_lr=0.1
# name="mango_v3_head_ns-lr${head_lr}"


# ... Bias
# ......
# attn_b_scale_dim=True
# vec_b_scale_dim=True
# name="mango_v3_bias_l2_dim"

# ......
# attn_b_normalize=null
# vec_b_normalize=null
# name="mango_v3_bias_null"


# ... Embedding
# ......
# embedding_normalize="l2_col"
# embedding_scale_dim=True
# name="mango_v3_emb_l2col-T_dim"
# # <next>
# embedding_scale_dim=False
# name="mango_v3_emb_l2col-T_nodim"

# ......
# embedding_normalize="l2_col"
# embedding_scale_dim_transpose=False
# name="mango_v3_emb_l2col"
# # <next>
# embedding_scale_dim=True
# name="mango_v3_emb_l2col_dim-scale"


# ... NEW EXPERIMENTS OF COUPLED NORMALIZATION
#     WITH GRAD_SQUARED PRECONDITIONING
# ...... Fixed beta2=0.95
# beta2=0.95
# coupled_normalize=True
# coupled_normalize_power=0.5
# coupled_normalize_correct_bias=True

# # lr=0.03
# # lr=0.01
# # lr=3e-3
# # lr=1e-3
# # lr=3e-4
# # lr=1e-4
# # lr=3e-5
# lr=1e-5
# # lr=3e-6
# # lr=1e-6
# # name="mango_v3_coupled_beta2${beta2}_p${coupled_normalize_power}_lr${lr}"
# use_amp=False
# name="mango_v3_coupled-no-amp_p${coupled_normalize_power}_lr${lr}"


# ... SCALE_BY_TENSOR_NORM EXPERIMENTS
# ...... Use relative norm scaling, scale only intermediate linear layers
# NOTE: since we are using relative scaling, we do not worry about dimension factors.
# mat_scale_norm="op"
# attn_w_scale_norm="op"
# scale_norm_power=1.0
# scale_norm_ratio=True
# scale_norm_clip_min=1e-4

# lr=0.01
# lr=0.03
# lr=1e-3
# lr=3e-3
# lr=1e-4

# name="mango_v3_scale_mat-attn_p${scale_norm_power}_ratio${scale_norm_ratio}_lr${lr}"


# ...... Relative norm scaling, scale only embedding layer
# NOTE: we do not transpose the matrix, so the 1->inf norm on original
# embedding matrix is just the row-max lr-norm.
# embedding_scale_norm="rowmax_l2"

# scale_norm_power=1.0
# scale_norm_ratio=True
# scale_norm_clip_min=1e-4

# lr=0.01
# lr=0.03
# lr=1e-3
# lr=3e-3
# lr=1e-4

# name="mango_v3_scale_emb_p${scale_norm_power}_ratio${scale_norm_ratio}_lr${lr}"


# ...... Relative norm scaling, scale only bias vectors.
# attn_b_scale_norm="l2"
# vec_b_scale_norm="l2"

# scale_norm_power=1.0
# scale_norm_ratio=True
# scale_norm_clip_min=1e-4

# lr=0.01
# lr=0.03
# lr=1e-3
# lr=3e-3
# lr=1e-4

# name="mango_v3_scale_bias_p${scale_norm_power}_ratio${scale_norm_ratio}_lr${lr}"


# ...... Relative norm scaling, scale only LayerNorm weights.
# vec_w_scale_norm="inf_"

# scale_norm_power=1.0
# scale_norm_ratio=True
# scale_norm_clip_min=1e-4

# lr=0.01
# lr=0.03
# lr=1e-3
# lr=3e-3
# lr=1e-4

# name="mango_v3_scale_vecW_p${scale_norm_power}_ratio${scale_norm_ratio}_lr${lr}"


# ...... Relative norm scaling, scale all layers with proper tensor norms.
# mat_scale_norm="op"
# attn_w_scale_norm="op"
# head_scale_norm="op"
# embedding_scale_norm="rowmax_l2"
# vec_w_scale_norm="inf_"
# attn_b_scale_norm="l2"
# vec_b_scale_norm="l2"

# scale_norm_power=1.0
# scale_norm_ratio=True
# scale_norm_clip_min=1e-4

# lr=0.01
# lr=0.03
# lr=1e-3
# lr=3e-3
# lr=1e-4

# name="mango_v3_scale_all_p${scale_norm_power}_ratio${scale_norm_ratio}_lr${lr}"


# ... San checks
# project="mango_v3_test_qinzi"
# # ......
# # name="baseline"
# # ......
# # head_lr=0.01
# # name="head_0.01"
# # ......
# head_normalize="ns"
# scale_dim=True
# # head_lr=0.01
# # name="head_0.01_new"
# head_lr=5e-5
# name="head_5e-5"


# ========================================================================
# Below is submit function. Only change the part related to global
# vs param-wise learning rate.
# ========================================================================

parse() {
    # $1: config string
    # $2: variable name
    if [ -n "${!2+x}" ]; then
        echo "$1=${!2}"
    fi
}

# project=null

layers=("mat" "embedding" "head" "attn_w" "attn_b" "vec_w" "vec_b")
keys=(
  # base
  "lr"
  "beta1"
  "beta2"
  "nesterov"
  "eps"
  # normalization
  "normalize"
  "scale_dim"
  "scale_dim_transpose"
  "scale_dim_clip_min"
  "scale_dim_clip_max"
  "ns_steps"
  "num_heads"
  # norm scaling
  "scale_norm"
  "scale_norm_ratio"
  "scale_norm_power"
  "scale_norm_clip_min"
  "scale_norm_clip_max"
  # others
  "use_adamw"
  "offset_beta"
  "igt_scale"
  "coupled_normalize"
  "coupled_normalize_power_pre"
  "coupled_normalize_power_post"
  "coupled_normalize_correct_bias"
)

# Start building the args list.
args=(
  "logging.wandb_project=$project"
  "logging.wandb_name=$name"
  "logging.log_callback_data=$log_data"
  "train.max_steps=$steps"
  "train.use_amp=$use_amp"
  "dataset.total_batch_size=$batch_size"
  "experimental=null"
  "test=null"
  "optimizer=$optimizer"
  # lr schedule
  "optimizer/lr_config=$schedule"
  "optimizer.lr_config.warmup=$warmup"
  "optimizer.lr_config.const=$const"
  "optimizer.lr_config.max_steps=$steps"
)

# Override default (global) configs
for layer in "${layers[@]}"; do
  for key in "${keys[@]}"; do
    args+=( "$(parse "optimizer.core.${layer}.${key}" "${key}")" )
  done
done

# Override layer-specific configs
# Layer-specific configs always have higher priority than global ones.
for layer in "${layers[@]}"; do
  for key in "${keys[@]}"; do
    args+=( "$(parse "optimizer.core.${layer}.${key}" "${layer}_${key}")" )
  done
done

# python main.py ${args[@]}

job_output=$(qsub <<EOF
#!/bin/bash -l

#$ -pe omp 8
#$ -l gpus=1
#$ -l gpu_type=L40S     # Specifies the gpu type.
#$ -l h_rt=8:00:00      # Specifies the hard time limit for the job
#$ -N "$name".sh
#$ -o $OUTPUT_PATH/\$JOB_NAME.o\$JOB_ID     # Escape environment variables with \$
#$ -e $OUTPUT_PATH/\$JOB_NAME.e\$JOB_ID

sleep $(((RANDOM % 1000) / 100))   # Prevents simultaneous reads of loadit dataset

source activate_env.sh
python main.py ${args[@]}
EOF
)

# Save job id and associated name to local .txt
# This is extremely helpful to manage a bunch of experiments.
job_id=$(echo "$job_output" | awk '{print $3}')
echo "job_id: ${job_id} || ${name}" >> "${OUTPUT_PATH}/job_list.txt"

echo "Submitted job: $name"