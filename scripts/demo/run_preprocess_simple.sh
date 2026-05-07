#!/bin/bash
# Simple parallel preprocessing launcher
# Runs 8 independent processes, each with its own GPU

TOTAL_GPUS=8
LAMBDA_DECAY=${1:-1.0}
SAMPLE_CT=${2:-32}

echo "Starting preprocessing on $TOTAL_GPUS GPUs..."
echo "Each process will handle ~$(( (1513 + TOTAL_GPUS - 1) / TOTAL_GPUS )) scenes"

# Launch processes in background, each with different GPU
for gpu_id in $(seq 0 $((TOTAL_GPUS - 1))); do
    echo "Launching GPU $gpu_id..."
    CUDA_VISIBLE_DEVICES=$gpu_id python scripts/demo/preprocess_scannet_simple.py \
        --gpu-id $gpu_id \
        --total-gpus $TOTAL_GPUS \
        --lambda-decay $LAMBDA_DECAY \
        --sample-ct $SAMPLE_CT \
        --model-path "Qwen/Qwen3-VL-8B-Instruct" \
        > logs/preprocess_gpu_${gpu_id}.log 2>&1 &
done

echo "All processes launched! Check logs in logs/preprocess_gpu_*.log"
echo "Monitor progress with: tail -f logs/preprocess_gpu_*.log"
echo ""
echo "Wait for all processes to complete..."
wait

echo "All preprocessing complete!"
