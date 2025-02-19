#!/bin/bash

# mango_v3b.
# What's new from mango_v3?
#   - Updates to frobenius normalization in newton_schulz method;
#   - Defaults to some normalization for all layers.
#   - *The optimizer itself remains the same. We only change configs.

optimizer=mango_v3

# ========================================================================
# Global Configs.
# ========================================================================

project=pile_baseline

log_data=False
steps=2000
batch_size=128

schedule=linear
warmup=200
const=null

# Make scc_outputs dir.
exp_name=mango_v3b
BASE_DIR=/projectnb/aclab/qinziz/trainit
DATE=$(date +"%Y-%m-%d")
OUTPUT_PATH=$BASE_DIR/scc_outputs/$DATE/$exp_name

mkdir -p $OUTPUT_PATH


# ========================================================================
# mango_v3b default optimzier configs
# ========================================================================

# Use 1->rms induced norm for embedding layers
embedding_normalize="l2_col"
embedding_scale_dim=True
# Use rms->rms induced norm for diag(vec_w), which is equivalent to Sign
# vec_w_normalize="inf_"
# NOTE: once we decided the right norm for head, we will update it here
# but for now, we just use the default rms->rms induced norm.


# ========================================================================
# Experiment configs
# ========================================================================

# ...
# name="mango_v3b_baseline"


# O. Mango baselines
# O.a. Baseline with larger lrs.
# lr=0.03
# bias_lr=0.01
# attn_b_lr=$bias_lr
# vec_b_lr=$bias_lr
# name="mango_v3b_lr${lr}_bias${bias_lr}"


# I. Tuning learning rates of each layer
# I.a. Embedding
# candidates=(0.03 3e-3 1e-3 3e-4)
# embedding_lr=${candidates[-1]}
# name="mango_v3b_emb-lr${embedding_lr}"

# I.b. Head
# candidates=(0.1 0.3 1.0)
# head_lr=${candidates[-1]}
# name="mango_v3b_head-lr${head_lr}"

# I.c. Linear weights
# candidates=(0.03 3e-3)
# mat_lr=${candidates[-1]}
# attn_w_lr=$mat_lr
# name="mango_v3b_linear-lr${mat_lr}"

# I.d. Bias vectors
# candidates=(0.03 3e-3 0.02)
# attn_b_lr=${candidates[-1]}
# vec_b_lr=${candidates[-1]}
# name="mango_v3b_bias-lr${attn_b_lr}"

# I.e. LayerNorm weights
# candidates=(0.03 3e-3)
# vec_w_lr=${candidates[-1]}
# name="mango_v3b_vecw-lr${vec_w_lr}"


# II. Testing coupled normalization (pre- and post-conditioning) of each layer
# II.0. San-check
#       We should recover the performance of adam-muon with coupled normalization.
offset_beta=0.0
coupled_normalize=True
embedding_normalize=null
head_normalize=null
attn_b_normalize=null
vec_b_normalize=null
lr=0.03

# name="mango_v3b_recovers_precmuon"



# ========================================================================
# Submit function
# ========================================================================

# project=null  # for testing only
parse() {
    # $1: config string
    # $2: variable name
    if [ -n "${!2+x}" ]; then
        echo "$1=${!2}"
    fi
}

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
#   "train.use_amp=$use_amp"
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