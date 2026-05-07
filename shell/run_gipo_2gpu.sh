#!/bin/bash
#SBATCH --job-name=gipo_2gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=logs/gipo_2gpu_%j.out
#SBATCH --error=logs/gipo_2gpu_%j.err
#SBATCH --partition=fengl2


MODEL=${1:-"qwen2.5-7b"}
STEPS=${2:-"3,4,5"}
STAGE=${3:-"grpo"}

ADAMACRO_DIR="/path/to/CIPO"
cd ${ADAMACRO_DIR}

mkdir -p logs

echo "============================================================"
echo "GIPO 2-GPU Experiment (Model Parallelism)"
echo "============================================================"
echo "Job ID:     ${SLURM_JOB_ID}"
echo "Node:       ${SLURM_NODELIST}"
echo "GPUs:       ${CUDA_VISIBLE_DEVICES}"
echo "Model:      ${MODEL}"
echo "Steps:      ${STEPS}"
echo "Stage:      ${STAGE}"
echo "Time:       $(date)"
echo "============================================================"

source $CONDA_PREFIX/etc/profile.d/conda.sh
conda activate tool

python scripts/run_pipeline_gipo_2gpu.py \
    --model ${MODEL} \
    --steps ${STEPS} \
    --stage ${STAGE} \
    --max-merges 50 \
    --min-freq 3 \
    --max-macro-len 6 \
    --max-episodes 100 \
    --max-turns 30 \
    --max-atomic-calls 50

echo "============================================================"
echo "GIPO 2-GPU experiment completed at $(date)"
echo "============================================================"
