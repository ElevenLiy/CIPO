#!/bin/bash
#SBATCH --job-name=fig1d_grpo
#SBATCH --partition=fengl2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/fig1d_grpo_%j.out
#SBATCH --error=logs/fig1d_grpo_%j.err


MODEL=${1:-"qwen2.5-7b"}

ADAMACRO_DIR="/path/to/CIPO"
cd ${ADAMACRO_DIR}

mkdir -p logs

SFT_RESULTS="../figure1/outputs/exp_c_full_sft.json"

if [ ! -f "${SFT_RESULTS}" ]; then
    echo "ERROR: SFT results not found at ${SFT_RESULTS}"
    echo "Run the SFT experiment first: sbatch run_figure1d_sft.sh"
    exit 1
fi

echo "============================================================"
echo "Figure 1(d): Counterfactual Rollout — Standard GRPO policy"
echo "============================================================"
echo "Job ID:     ${SLURM_JOB_ID}"
echo "Node:       ${SLURM_NODELIST}"
echo "GPU:        ${CUDA_VISIBLE_DEVICES}"
echo "Model:      ${MODEL}"
echo "Policy:     standard_grpo"
echo "Replay:     ${SFT_RESULTS}"
echo "Time:       $(date)"
echo "============================================================"

source $CONDA_PREFIX/etc/profile.d/conda.sh
conda activate tool

python scripts/run_figure1d_rollout.py \
    --model ${MODEL} \
    --policy_name standard_grpo \
    --lora_path outputs/TOOLATHLON/checkpoints/standard_grpo/${MODEL} \
    --replay_from ${SFT_RESULTS} \
    --seed 42

echo "============================================================"
echo "Done at $(date)"
echo "============================================================"
