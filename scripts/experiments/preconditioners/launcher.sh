#!/bin/bash

# Experiment launcher (IN PROGRESS)

project=pile_baseline
exp_name="mango_v3"

BASE_DIR=/projectnb/aclab/qinziz/trainit
DATE=$(date +"%Y-%m-%d")
OUTPUT_PATH=$BASE_DIR/scc_outputs/$DATE/$exp_name

mkdir -p $OUTPUT_PATH


# ========================================================================
# Mango optimzier configs
# ========================================================================

# ... NEW EXPERIMENTS OF COUPLED NORMALIZATION
#     WITH GRAD_SQUARED PRECONDITIONING
# ...... Fixed beta2=0.95
# coupled_normalize=True
# coupled_normalize_power=0.5
# coupled_normalize_power=0.25
# name="mango_v3_coupled_normalize_p${coupled_normalize_power}"

# ...... Different beta2
coupled_normalize=True
coupled_normalize_power=0.5
coupled_normalize_power=0.25
beta2=0.99
name="mango_v3_coupled_normalize_beta2${beta2}_p${coupled_normalize_power}"


