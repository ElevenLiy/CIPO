#!/bin/bash
#SBATCH --job-name=traject_gipo_llama3b
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/traject_gipo_llama3b_%j.out
#SBATCH --error=logs/traject_gipo_llama3b_%j.err
#SBATCH --partition=fengl2


MODEL="llama3.2-3b"
STEPS=${1:-"3,4,5"}
STAGE=${2:-"grpo"}

ADAMACRO_DIR="/path/to/CIPO"
DATA_DIR="/path/to/CIPO/Traject-bench"
OUTPUT_DIR="/path/to/CIPO/outputs/traject"

cd ${ADAMACRO_DIR}
mkdir -p logs

echo "============================================================"
echo "Traject-bench — GIPO — LLaMA 3.2-3B (Single GPU)"
echo "============================================================"
echo "Job ID: ${SLURM_JOB_ID} | Node: ${SLURM_NODELIST} | GPU: ${CUDA_VISIBLE_DEVICES}"
echo "Model: ${MODEL} | Steps: ${STEPS} | Stage: ${STAGE}"
echo "Data: ${DATA_DIR} | Output: ${OUTPUT_DIR}"
echo "Time: $(date)"
echo "============================================================"

source $CONDA_PREFIX/etc/profile.d/conda.sh
conda activate tool

python scripts/run_pipeline_gipo_dataset.py \
    --dataset traject \
    --rl-dataset ${DATA_DIR}/rl_dataset_llm.json \
    --all-tools ${DATA_DIR}/all_tools.json \
    --tool-simulator-db ${DATA_DIR}/tool_simulator_database.json \
    --output-dir ${OUTPUT_DIR} \
    --model ${MODEL} \
    --steps ${STEPS} \
    --stage ${STAGE} \
    --gpu-mode 1gpu \
    --max-merges 50 --min-freq 3 --max-macro-len 6 \
    --max-episodes 100 --max-turns 30 --max-atomic-calls 50

echo "============================================================"
echo "Traject-bench — LLaMA 3.2-3B completed at $(date)"
echo "============================================================"
