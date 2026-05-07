#!/bin/bash
# Simple parallel preprocessing launcher for robofac video dataset
# Runs 8 independent processes, each with its own GPU

TOTAL_GPUS=8
SAMPLE_CT=${1:-32}

echo "Starting robofac preprocessing on $TOTAL_GPUS GPUs..."

mkdir -p logs

# Launch processes in background, each with different GPU
for gpu_id in $(seq 0 $((TOTAL_GPUS - 1))); do
    echo "Launching GPU $gpu_id..."
    CUDA_VISIBLE_DEVICES=$gpu_id python scripts/demo/preprocess_robofac_simple.py \
        --gpu-id $gpu_id \
        --total-gpus $TOTAL_GPUS \
        --sample-ct $SAMPLE_CT \
        --model-path "Qwen/Qwen3-VL-4B-Instruct" \
        > logs/preprocess_robofac_gpu_${gpu_id}.log 2>&1 &
done

echo "All processes launched! Check logs in logs/preprocess_robofac_gpu_*.log"
echo "Monitor progress with: tail -f logs/preprocess_robofac_gpu_*.log"
echo ""
echo "Wait for all processes to complete..."
wait

echo "All preprocessing complete!"
