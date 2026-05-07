#!/bin/bash
#SBATCH --job-name=gipo
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/gipo_%j.out
#SBATCH --error=logs/gipo_%j.err
#SBATCH --partition=fengl2


MODEL=${1:-"qwen2.5-7b"}
STEPS=${2:-"1,2,3,4,5"}
STAGE=${3:-"grpo"}

ADAMACRO_DIR="/path/to/CIPO"
cd ${ADAMACRO_DIR}

mkdir -p logs

echo "============================================================"
echo "GIPO Experiment"
echo "============================================================"
echo "Job ID:     ${SLURM_JOB_ID}"
echo "Node:       ${SLURM_NODELIST}"
echo "GPU:        ${CUDA_VISIBLE_DEVICES}"
echo "Model:      ${MODEL}"
echo "Steps:      ${STEPS}"
echo "Stage:      ${STAGE}"
echo "Time:       $(date)"
echo "============================================================"

source $CONDA_PREFIX/etc/profile.d/conda.sh
conda activate tool

python scripts/run_pipeline_gipo.py \
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
echo "GIPO experiment completed at $(date)"
echo "============================================================"
