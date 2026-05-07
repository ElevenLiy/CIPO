#!/bin/bash
#SBATCH --job-name=adamacro
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/adamacro_%j.out
#SBATCH --error=logs/adamacro_%j.err 
#SBATCH --partition=fengl2


MODEL=${1:-"qwen2.5-7b"}
STEPS=${2:-"1,2,3,4,5"}
STAGE=${3:-"grpo"}
EPOCHS=${4:-""}
GROUP_SIZE=${5:-""}

ADAMACRO_DIR="/path/to/CIPO"
cd ${ADAMACRO_DIR}

mkdir -p logs

echo "============================================================"
echo "AdaMacro Experiment"
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


EXTRA_ARGS=""
if [ -n "${EPOCHS}" ]; then
    EXTRA_ARGS="${EXTRA_ARGS} --epochs ${EPOCHS}"
fi
if [ -n "${GROUP_SIZE}" ]; then
    EXTRA_ARGS="${EXTRA_ARGS} --group-size ${GROUP_SIZE}"
fi

python scripts/run_pipeline.py \
    --model ${MODEL} \
    --steps ${STEPS} \
    --stage ${STAGE} \
    --max-merges 50 \
    --min-freq 3 \
    --max-macro-len 6 \
    --max-episodes 100 \
    --max-turns 30 \
    --max-atomic-calls 50 \
    ${EXTRA_ARGS}

echo "============================================================"
echo "Experiment completed at $(date)"
echo "============================================================"
