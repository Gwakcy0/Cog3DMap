#!/bin/bash
# =============================================================================
# Unified Point3R-LLM Training Script
# =============================================================================
# Configure via environment variables before calling this script.
# Only EXP_NAME is required; all others have sensible defaults.
#
# Usage:
#   export EXP_NAME="my_experiment"
#   export MODEL_PATH="Qwen/Qwen3-VL-8B-Instruct"
#   bash scripts/train/train.sh
#
# For a full list of configurable parameters, see the "Defaults" section below.
# =============================================================================

set -e

# =============================================================================
# Defaults
# =============================================================================

# --- Required ---
if [[ -z "$EXP_NAME" ]]; then
    echo "ERROR: EXP_NAME must be set." >&2
    exit 1
fi

# --- Model ---
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-VL-8B-Instruct}"
DATASETS="${DATASETS:-scan2cap_point3r}"

# --- Memory Features ---
USE_POINTER_MEMORY="${USE_POINTER_MEMORY:-True}"
USE_PREPROCESSED_INPUT="${USE_PREPROCESSED_INPUT:-True}"
MERGE_MEMORY_FEAT="${MERGE_MEMORY_FEAT:-True}"
MEMORY_FUSION_METHOD="${MEMORY_FUSION_METHOD:-add}"
TUNE_MEMORY_FEATURE_PROJECTOR="${TUNE_MEMORY_FEATURE_PROJECTOR:-True}"
TUNE_MEMORY_FEATURE_FUSION="${TUNE_MEMORY_FEATURE_FUSION:-True}"

# --- RoPE (optional, only appended when non-empty) ---
ROPE_MODE="${ROPE_MODE:-}"
ROPE_POSITION_RANGE="${ROPE_POSITION_RANGE:-}"
USE_POINTER_POSITION_ENCODING="${USE_POINTER_POSITION_ENCODING:-}"
TUNE_ROPE3D_CONTINUOUS="${TUNE_ROPE3D_CONTINUOUS:-}"
TUNE_POINTER_POSITION_ENCODER="${TUNE_POINTER_POSITION_ENCODER:-}"

# --- Pointer Format (optional) ---
POINTER_FORMAT="${POINTER_FORMAT:-}"

# --- Pointer Data Directory Override (optional) ---
POINTER_DIR_NAME="${POINTER_DIR_NAME:-}"

# --- Frame ID labels for pointer tokens (optional) ---
ADD_FRAME_ID="${ADD_FRAME_ID:-}"

# --- Training Hyperparameters ---
TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:-16}"
LR="${LR:-1e-5}"
MM_PROJECTOR_LR="${MM_PROJECTOR_LR:-1e-5}"
VISION_TOWER_LR="${VISION_TOWER_LR:-1e-6}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
SAVE_STEPS="${SAVE_STEPS:-1000}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-scripts/zero2_opt.json}"
MODEL_MAX_LENGTH="${MODEL_MAX_LENGTH:-12800}"
SEED="${SEED:-0}"

# --- Tuning Flags ---
TUNE_MM_LLM="${TUNE_MM_LLM:-True}"
TUNE_MM_VISION="${TUNE_MM_VISION:-False}"
TUNE_MM_MLP="${TUNE_MM_MLP:-False}"

# --- Extra args passthrough ---
EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS:-}"

# =============================================================================
# Distributed Configuration
# =============================================================================
MASTER_ADDR="127.0.0.1"
MASTER_PORT=$(shuf -i 20000-29999 -n 1)
NPROC_PER_NODE=$(nvidia-smi --list-gpus | wc -l)
GRADIENT_ACCUMULATION_STEPS=$(($TOTAL_BATCH_SIZE / $NPROC_PER_NODE))

# =============================================================================
# Paths
# =============================================================================
OUTPUT_DIR="./outputs/${EXP_NAME}"
CACHE_DIR="./cache"
mkdir -p "$OUTPUT_DIR"

# =============================================================================
# WANDB
# =============================================================================
export NCCL_NVLS_ENABLE=0
export WANDB_PROJECT="${WANDB_PROJECT:-cog3dmap}"
export WANDB_RUN_NAME="${EXP_NAME}"

# =============================================================================
# Save for Reproducibility
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cp "${SCRIPT_DIR}/train.sh" "${OUTPUT_DIR}/train.sh"

cat > "${OUTPUT_DIR}/train_resolved.sh" << RESOLVED_EOF
#!/bin/bash
# Auto-generated for reproducibility
# Experiment: ${EXP_NAME}
# Generated: $(date "+%Y-%m-%d %H:%M:%S")
#
# To reproduce: bash ${OUTPUT_DIR}/train_resolved.sh

export EXP_NAME="${EXP_NAME}"
export MODEL_PATH="${MODEL_PATH}"
export DATASETS="${DATASETS}"
export USE_POINTER_MEMORY="${USE_POINTER_MEMORY}"
export USE_PREPROCESSED_INPUT="${USE_PREPROCESSED_INPUT}"
export MERGE_MEMORY_FEAT="${MERGE_MEMORY_FEAT}"
export MEMORY_FUSION_METHOD="${MEMORY_FUSION_METHOD}"
export TUNE_MEMORY_FEATURE_PROJECTOR="${TUNE_MEMORY_FEATURE_PROJECTOR}"
export TUNE_MEMORY_FEATURE_FUSION="${TUNE_MEMORY_FEATURE_FUSION}"
export ROPE_MODE="${ROPE_MODE}"
export ROPE_POSITION_RANGE="${ROPE_POSITION_RANGE}"
export USE_POINTER_POSITION_ENCODING="${USE_POINTER_POSITION_ENCODING}"
export TUNE_ROPE3D_CONTINUOUS="${TUNE_ROPE3D_CONTINUOUS}"
export TUNE_POINTER_POSITION_ENCODER="${TUNE_POINTER_POSITION_ENCODER}"
export POINTER_FORMAT="${POINTER_FORMAT}"
export POINTER_DIR_NAME="${POINTER_DIR_NAME}"
export ADD_FRAME_ID="${ADD_FRAME_ID}"
export TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE}"
export LR="${LR}"
export MM_PROJECTOR_LR="${MM_PROJECTOR_LR}"
export VISION_TOWER_LR="${VISION_TOWER_LR}"
export NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS}"
export SAVE_STEPS="${SAVE_STEPS}"
export DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG}"
export MODEL_MAX_LENGTH="${MODEL_MAX_LENGTH}"
export SEED="${SEED}"
export TUNE_MM_LLM="${TUNE_MM_LLM}"
export TUNE_MM_VISION="${TUNE_MM_VISION}"
export TUNE_MM_MLP="${TUNE_MM_MLP}"
export EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS}"

bash scripts/train/train.sh
RESOLVED_EOF
chmod +x "${OUTPUT_DIR}/train_resolved.sh"

# =============================================================================
# Build Command
# =============================================================================
CMD=(
    torchrun
    --nproc_per_node=$NPROC_PER_NODE
    --master_addr=$MASTER_ADDR
    --master_port=$MASTER_PORT
    src/qwen_vl/train/train_qwen.py
    --model_name_or_path "$MODEL_PATH"
    --tune_mm_llm "$TUNE_MM_LLM"
    --tune_mm_vision "$TUNE_MM_VISION"
    --tune_mm_mlp "$TUNE_MM_MLP"
    --dataset_use "$DATASETS"
    --output_dir "$OUTPUT_DIR"
    --cache_dir "$CACHE_DIR"
    --bf16
    --per_device_train_batch_size 1
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS
    --learning_rate "$LR"
    --mm_projector_lr "$MM_PROJECTOR_LR"
    --vision_tower_lr "$VISION_TOWER_LR"
    --optim adamw_torch
    --model_max_length "$MODEL_MAX_LENGTH"
    --data_flatten False
    --max_pixels $((576*28*28))
    --min_pixels $((16*28*28))
    --base_interval 2
    --video_max_frames 8
    --video_min_frames 4
    --video_max_frame_pixels $((1664*28*28))
    --video_min_frame_pixels $((256*28*28))
    --num_train_epochs "$NUM_TRAIN_EPOCHS"
    --warmup_ratio 0.03
    --lr_scheduler_type cosine
    --weight_decay 0.01
    --logging_steps 10
    --save_steps "$SAVE_STEPS"
    --save_total_limit 1
    --deepspeed "$DEEPSPEED_CONFIG"
    --gradient_checkpointing
    --dataloader_num_workers 4
    --group_by_modality_length true
    --seed "$SEED"
    --run_name "$EXP_NAME"
    --report_to wandb
    --use_pointer_memory "$USE_POINTER_MEMORY"
    --use_preprocessed_input "$USE_PREPROCESSED_INPUT"
    --merge_memory_feat "$MERGE_MEMORY_FEAT"
    --memory_fusion_method "$MEMORY_FUSION_METHOD"
    --tune_memory_feature_projector "$TUNE_MEMORY_FEATURE_PROJECTOR"
    --tune_memory_feature_fusion "$TUNE_MEMORY_FEATURE_FUSION"
)

# --- Conditional args (only appended when non-empty) ---
[[ -n "$ROPE_MODE" ]]                    && CMD+=(--rope_mode "$ROPE_MODE")
[[ -n "$ROPE_POSITION_RANGE" ]]          && CMD+=(--rope_position_range "$ROPE_POSITION_RANGE")
[[ -n "$USE_POINTER_POSITION_ENCODING" ]] && CMD+=(--use_pointer_position_encoding "$USE_POINTER_POSITION_ENCODING")
[[ -n "$TUNE_ROPE3D_CONTINUOUS" ]]        && CMD+=(--tune_rope3d_continuous "$TUNE_ROPE3D_CONTINUOUS")
[[ -n "$TUNE_POINTER_POSITION_ENCODER" ]] && CMD+=(--tune_pointer_position_encoder "$TUNE_POINTER_POSITION_ENCODER")
[[ -n "$POINTER_FORMAT" ]]                && CMD+=(--pointer_format "$POINTER_FORMAT")
[[ -n "$POINTER_DIR_NAME" ]]             && CMD+=(--pointer_dir_name "$POINTER_DIR_NAME")
[[ -n "$ADD_FRAME_ID" ]]                 && CMD+=(--add_frame_id "$ADD_FRAME_ID")

# --- Extra args passthrough ---
if [[ -n "$EXTRA_TRAIN_ARGS" ]]; then
    read -ra EXTRA_ARGS <<< "$EXTRA_TRAIN_ARGS"
    CMD+=("${EXTRA_ARGS[@]}")
fi

# =============================================================================
# Execute
# =============================================================================
echo "============================================="
echo "Training: ${EXP_NAME}"
echo "Model:    ${MODEL_PATH}"
echo "Dataset:  ${DATASETS}"
echo "Output:   ${OUTPUT_DIR}"
echo "GPUs:     ${NPROC_PER_NODE}"
echo "============================================="

if [[ "${DRY_RUN:-}" == "1" ]]; then
    echo "[DRY RUN] Command:"
    printf '%q ' "${CMD[@]}"
    echo ""
else
    "${CMD[@]}" 2>&1 | tee "${OUTPUT_DIR}/train.log"
    echo "============================================="
    echo "Training completed!"
    echo "Output saved to: ${OUTPUT_DIR}"
    echo "============================================="
fi
