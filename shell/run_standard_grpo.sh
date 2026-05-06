#!/bin/bash
#SBATCH --job-name=std_grpo
#SBATCH --partition=fengl2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/std_grpo_%j.out
#SBATCH --error=logs/std_grpo_%j.err

# ===========================================================================
# Standard GRPO Training (R = R_task only, no skill bias)
# ===========================================================================
# For Figure 1(d) baseline comparison.
#
# Usage:
#   sbatch run_standard_grpo.sh                        # Default: qwen2.5-7b, 1 epoch, G=2
#   sbatch run_standard_grpo.sh qwen2.5-7b 1 2        # Explicit
#   sbatch run_standard_grpo.sh qwen2.5-7b 2 4        # 2 epochs, G=4
# ===========================================================================

MODEL=${1:-"qwen2.5-7b"}
EPOCHS=${2:-1}
GROUP_SIZE=${3:-2}

ADAMACRO_DIR="/path/to/CIPO"
cd ${ADAMACRO_DIR}

mkdir -p logs

echo "============================================================"
echo "Standard GRPO Training (R = R_task only)"
echo "============================================================"
echo "Job ID:     ${SLURM_JOB_ID}"
echo "Node:       ${SLURM_NODELIST}"
echo "GPU:        ${CUDA_VISIBLE_DEVICES}"
echo "Model:      ${MODEL}"
echo "Epochs:     ${EPOCHS}"
echo "Group size: ${GROUP_SIZE}"
echo "Time:       $(date)"
echo "============================================================"

source $CONDA_PREFIX/etc/profile.d/conda.sh
conda activate tool

python scripts/step4_standard_grpo.py \
    --model ${MODEL} \
    --epochs ${EPOCHS} \
    --group-size ${GROUP_SIZE} \
    --seed 42

echo "============================================================"
echo "Done at $(date)"
echo "============================================================"
