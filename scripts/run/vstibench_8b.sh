#!/bin/bash
#SBATCH --job-name=Point3R-LLM-vstibench_8b_train_n_eval
#SBATCH -o sbatch_log/vstibench-8b-32frame-video.%j.out
#SBATCH --partition=<your_partition>  # TODO: set to your cluster's partition
#SBATCH --gres=gpu:8
#SBATCH --nodes=1

# Experiment: VSTIBench 8B with memory features (32-frame video)

source venv/bin/activate

export EXP_NAME="vstibench_Qwen3VL_8b_memfeat_32frame_video"
export MODEL_PATH="Qwen/Qwen3-VL-8B-Instruct"
export DATASETS="vstibench_point3r"

# Memory features
export MERGE_MEMORY_FEAT="True"
export MEMORY_FUSION_METHOD="add"

# Frame id 
export ADD_FRAME_ID="True"

# Evaluation
export BENCHMARKS="vstibench_point3r"
export EVAL_MODEL_TYPE="point3r_llm_v2"
export LOG_SUFFIX="vstibench_eval"

# --- Train ---
bash scripts/train/train.sh

# --- Evaluate ---
export MODEL_PATH="./outputs/${EXP_NAME}"
bash scripts/evaluation/eval.sh

nvidia-smi
date
squeue --job $SLURM_JOBID
echo "##### END #####"
