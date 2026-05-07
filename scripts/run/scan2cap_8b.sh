#!/bin/bash
#SBATCH --job-name=Point3R-LLM-8b_memfeat
#SBATCH -o sbatch_log/8b-memfeat.%j.out
#SBATCH --partition=<your_partition>  # TODO: set to your cluster's partition
#SBATCH --gres=gpu:8
#SBATCH --nodes=1

# Experiment: 8B with memory features on Scan2Cap

source venv/bin/activate

export EXP_NAME="scan2cap_point3r_Qwen3VL_8b_memfeat"
export MODEL_PATH="Qwen/Qwen3-VL-8B-Instruct"
export DATASETS="scan2cap_point3r"

# Memory features
export MERGE_MEMORY_FEAT="True"
export MEMORY_FUSION_METHOD="add"

# Evaluation
export BENCHMARKS="scan2cap_point3r"
export EVAL_MODEL_TYPE="point3r_llm_v2"

# --- Train ---
bash scripts/train/train.sh

# --- Evaluate ---
export MODEL_PATH="./outputs/${EXP_NAME}"
bash scripts/evaluation/eval.sh

nvidia-smi
date
squeue --job $SLURM_JOBID
echo "##### END #####"
