#!/bin/bash
# Muon baseline.

optimizer=muon

# ========================================================================
# Global Configs.
# ========================================================================
project=check_muon #pile_baseline

log_data=False
steps=2000
batch_size=128

schedule=linear
warmup=200
const=null

postcondition=True
beta2=0.0

# Make scc_outputs dir.
exp_name=muon
BASE_DIR=/projectnb/aclab/cutkosky/qinzitrainit/trainit/
DATE=$(date +"%Y-%m-%d")
OUTPUT_PATH=$BASE_DIR/scc_outputs/$DATE/$exp_name

mkdir -p $OUTPUT_PATH


# ========================================================================
# Optimizer Configs.
# ========================================================================

lr=0.03
adam_lr=0.03
name="muon_post${postcondition}_beta2${beta2}_lr${lr}_adam-lr_${adam_lr}"


# ========================================================================
# Submit function.
# ========================================================================
parse() {
    # $1: config string
    # $2: variable name
    if [ -n "${!2+x}" ]; then
        echo "$1=${!2}"
    fi
}

# project=null    # test purpose
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
    "$(parse "optimizer.lr_config.lr" "lr")"
)

optimizer_keys=(
    "momentum"
    "beta2"
    "nesterov"
    "ns_steps"
    "adam_lr"
    "adam_beta1"
    "adam_beta2"
    "adam_eps"
    "adam_wd"
    "postcondition"
)

# Update optimizer configs
for key in "${optimizer_keys[@]}"; do
    args+=( "$(parse "optimizer.${key}" "${key}")" )
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
