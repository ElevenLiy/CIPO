#!/bin/bash
#SBATCH --job-name=toucan_gipo_qwen1.5b
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/toucan_gipo_qwen1.5b_%j.out
#SBATCH --error=logs/toucan_gipo_qwen1.5b_%j.err
#SBATCH --partition=fengl2

# ===========================================================================
# Toucan Dataset — GIPO Single-GPU — Qwen2.5-1.5B
# ===========================================================================
# Checkpoint: outputs/toucan/checkpoints/gipo_qwen25_15b/
# Eval:       outputs/toucan/eval_gipo_qwen25_15b_results/
#
# Usage:
#   sbatch toucan_run_gipo_qwen1.5b.sh                    # Default: steps 3,4,5
#   sbatch toucan_run_gipo_qwen1.5b.sh 1,2,3,4,5          # Full pipeline
#   sbatch toucan_run_gipo_qwen1.5b.sh 4,5                # GIPO + eval only
#   sbatch toucan_run_gipo_qwen1.5b.sh 5 sft              # Eval SFT only
# ===========================================================================

MODEL="qwen2.5-1.5b"
STEPS=${1:-"3,4,5"}
STAGE=${2:-"grpo"}

# --- Project paths ---
ADAMACRO_DIR="/path/to/CIPO"
DATA_DIR="/path/to/CIPO/Toucan"
OUTPUT_DIR="/path/to/CIPO/outputs/toucan"

cd ${ADAMACRO_DIR}
mkdir -p logs

echo "============================================================"
echo "Toucan — GIPO — Qwen2.5-1.5B (Single GPU)"
echo "============================================================"
echo "Job ID:     ${SLURM_JOB_ID}"
echo "Node:       ${SLURM_NODELIST}"
echo "GPU:        ${CUDA_VISIBLE_DEVICES}"
echo "Model:      ${MODEL}"
echo "Steps:      ${STEPS}"
echo "Stage:      ${STAGE}"
echo "Data:       ${DATA_DIR}"
echo "Output:     ${OUTPUT_DIR}"
echo "Time:       $(date)"
echo "============================================================"

# --- Environment ---
source $CONDA_PREFIX/etc/profile.d/conda.sh
conda activate tool

# --- Run ---
python scripts/run_pipeline_gipo_dataset.py \
    --dataset toucan \
    --rl-dataset ${DATA_DIR}/rl_dataset_toucan.json \
    --all-tools ${DATA_DIR}/all_tools_toucan.json \
    --tool-simulator-db ${DATA_DIR}/tool_simulator_database_toucan.json \
    --output-dir ${OUTPUT_DIR} \
    --model ${MODEL} \
    --steps ${STEPS} \
    --stage ${STAGE} \
    --gpu-mode 1gpu \
    --max-merges 50 \
    --min-freq 3 \
    --max-macro-len 6 \
    --max-episodes 100 \
    --max-turns 30 \
    --max-atomic-calls 50

echo "============================================================"
echo "Toucan — Qwen2.5-1.5B completed at $(date)"
echo "============================================================"
