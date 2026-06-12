#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# GPUs, processes, and torchrun port.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-$(echo "${CUDA_VISIBLE_DEVICES}" | awk -F',' '{print NF}')}"
MASTER_PORT="${MASTER_PORT:-29503}"

# Common runtime environment.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export HF_HOME="${HF_HOME:-D:/HF_HOME}"
# export NCCL_DEBUG=INFO
# export NCCL_P2P_DISABLE=1
# export NCCL_IB_DISABLE=1

# Paths. Override these from the command line environment when needed.
MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/model/Qwen3.6-4B}"
DEFAULT_DATA_PATH="${PROJECT_ROOT}/processed_dataset/contextual_dpo_tokenized_arrow"
if [[ ! -d "${DEFAULT_DATA_PATH}" && -d "${PROJECT_ROOT}/dataset/dpo/.cache/dpo_tokenized_arrow_verify" ]]; then
    DEFAULT_DATA_PATH="${PROJECT_ROOT}/dataset/dpo/.cache/dpo_tokenized_arrow_verify"
fi
DATA_PATH="${DATA_PATH:-${DEFAULT_DATA_PATH}}"
OUT_DIR="${OUT_DIR:-${PROJECT_ROOT}/model/dpo_fsdp}"

echo "============================================================"
echo "PROJECT_ROOT         = ${PROJECT_ROOT}"
echo "CUDA_VISIBLE_DEVICES = ${CUDA_VISIBLE_DEVICES}"
echo "NPROC_PER_NODE       = ${NPROC_PER_NODE}"
echo "MASTER_PORT          = ${MASTER_PORT}"
echo "MODEL_PATH           = ${MODEL_PATH}"
echo "DATA_PATH            = ${DATA_PATH}"
echo "OUT_DIR              = ${OUT_DIR}"
echo "============================================================"

torchrun \
    --standalone \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --master_port="${MASTER_PORT}" \
    -m dpo.main \
    --model_path "${MODEL_PATH}" \
    --data_path "${DATA_PATH}" \
    --out_dir "${OUT_DIR}" \
    --epochs 1 \
    --batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 5e-7 \
    --beta 0.1 \
    --max_len 4096 \
    --dtype bfloat16 \
    --grad_clip 1.0 \
    --warmup_iters 100 \
    --weight_decay 0.1 \
    --betas 0.9 0.95 \
    --fsdp_sharding_strategy full_shard \
    `#--gradient_checkpointing` \
    --num_workers 4 \
    --log_interval 10 \
    --save_interval 500 \
    --snapshot_interval 0 \
    "$@"
