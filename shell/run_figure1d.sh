#!/bin/bash
#SBATCH --job-name=fig1d
#SBATCH --partition=fengl2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/fig1d_%j.out
#SBATCH --error=logs/fig1d_%j.err

# ===========================================================================
# Figure 1 Panel (d): Counterfactual rollout experiment
# ===========================================================================
# Now runs 2 base rollouts per prompt (normal + skill-biased), branching at
# ALL eligible steps — yields ~5-10x more divergence states than before.
#
# Usage:
#   sbatch run_figure1d.sh                         # Default: qwen2.5-7b, 200 prompts
#   sbatch run_figure1d.sh qwen2.5-7b 200         # Explicit
#   sbatch run_figure1d.sh qwen2.5-1.5b 300       # 1.5B model, more prompts
# ===========================================================================

MODEL=${1:-"qwen2.5-7b"}
N_PROMPTS=${2:-200}

ADAMACRO_DIR="/path/to/CIPO"
cd ${ADAMACRO_DIR}

mkdir -p logs

echo "============================================================"
echo "Figure 1(d): Counterfactual Rollout"
echo "============================================================"
echo "Job ID:     ${SLURM_JOB_ID}"
echo "Node:       ${SLURM_NODELIST}"
echo "GPU:        ${CUDA_VISIBLE_DEVICES}"
echo "Model:      ${MODEL}"
echo "N prompts:  ${N_PROMPTS}"
echo "Time:       $(date)"
echo "============================================================"

source $CONDA_PREFIX/etc/profile.d/conda.sh
conda activate tool

python scripts/run_figure1d_rollout.py \
    --model ${MODEL} \
    --n-prompts ${N_PROMPTS} \
    --seed 42

echo "============================================================"
echo "Done at $(date)"
echo "============================================================"
