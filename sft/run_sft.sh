#!/usr/bin/env bash
set -euo pipefail

# 切到项目根目录（脚本所在目录的上一级），因为 main.py 内部使用了相对导入
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# ===== 可配置的环境变量 =====
# 使用的 GPU 卡号，例：CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
# nproc_per_node 默认等于 CUDA_VISIBLE_DEVICES 中的卡数
NPROC_PER_NODE="${NPROC_PER_NODE:-$(echo "${CUDA_VISIBLE_DEVICES}" | awk -F',' '{print NF}')}"
# 单机多卡端口（跟 pretrain 用不同端口，便于同机器并行跑两个阶段）
MASTER_PORT="${MASTER_PORT:-29501}"

# NCCL / tokenizers 常用环境变量
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
# export NCCL_DEBUG=INFO
# export NCCL_P2P_DISABLE=1
# export NCCL_IB_DISABLE=1

echo "============================================================"
echo "PROJECT_ROOT         = ${PROJECT_ROOT}"
echo "CUDA_VISIBLE_DEVICES = ${CUDA_VISIBLE_DEVICES}"
echo "NPROC_PER_NODE       = ${NPROC_PER_NODE}"
echo "MASTER_PORT          = ${MASTER_PORT}"
echo "============================================================"

# ===== 训练参数（按需修改）=====
# 额外参数可从命令行追加，会原样透传给 main.py
# 例：bash sft/run_sft.sh --epochs 2 --batch_size 16 --use_swanlab
torchrun \
    --standalone \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --master_port="${MASTER_PORT}" \
    -m sft.main \
    `# ===== 加载与输出 =====` \
    `# 起点：pretrain 输出的 HF 目录` \
    --load_dir /root/autodl-tmp/MyLLMDataset/base_model/hf_model_291M \
    --out_dir  /root/autodl-tmp/MyLLMDataset/sft_model \
    `# ===== 基础训练参数 =====` \
    --epochs 3 \
    --batch_size 32 \
    `# SFT 学习率通常比 pretrain 小 10×（pretrain 3e-4 → SFT 2e-5~5e-5）` \
    --learning_rate 2e-5 \
    --dtype bfloat16 \
    `# ===== 实验跟踪与数据 =====` \
    --num_workers 4 \
    --data_path /root/autodl-tmp/MyLLMDataset/processed_dataset/BelleGroup_sft_tokenized.jsonl \
    `# 启用 SwanLab 取消下一行注释（或命令行追加 --use_swanlab）` \
    --use_swanlab \
    `# ===== 训练优化 =====` \
    `# SFT 数据少，gradient_accumulation_steps 一般小一些` \
    --gradient_accumulation_steps 2 \
    --gradient_checkpointing \
    --grad_clip 1.0 \
    `# SFT 总步数远少于 pretrain，warmup 也按比例减小` \
    --warmup_iters 100 \
    --weight_decay 0.1 \
    --betas 0.9 0.95 \
    `# ===== 日志与保存 =====` \
    --log_interval 20 \
    --save_interval 500 \
    --snapshot_interval 2000 \
    `# ===== HF 加载方式（默认走 Transformer.from_pretrained；想用 AutoModelForCausalLM 加 --use_auto） =====` \
    `# --use_auto` \
    `# ===== 断点续训（默认全新 SFT；续训时取消下一行并改路径） =====` \
    `# --resume /root/autodl-tmp/MyLLMDataset/sft_model/sft_param_count291.550M.pt` \
    "$@"
