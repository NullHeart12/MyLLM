#!/usr/bin/env bash
set -euo pipefail

# 切到项目根目录（脚本所在目录的上一级），因为 main.py 内部使用了相对导入
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# ===== 可配置的环境变量 =====
# 使用的 GPU 卡号；QLoRA 27B 在 40G 上通常单卡足够（FSDP 才需要多卡）
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
# nproc_per_node 默认等于 CUDA_VISIBLE_DEVICES 中的卡数
NPROC_PER_NODE="${NPROC_PER_NODE:-$(echo "${CUDA_VISIBLE_DEVICES}" | awk -F',' '{print NF}')}"
# 跟 pretrain (29500) / sft (29501) 错开端口，便于同机器并行跑多个阶段
MASTER_PORT="${MASTER_PORT:-29502}"

# NCCL / tokenizers 常用环境变量
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
# HF Hub 镜像（如果从 HF 下 base 而非 modelscope，取消下一行）
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
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
# 例：bash lora/run_lora.sh --epochs 2 --lora_rank 16
torchrun \
    --standalone \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --master_port="${MASTER_PORT}" \
    -m lora.qlora_main \
    `# ===== 加载与输出 =====` \
    `# QLoRA 起点：HF Hub 名字或本地 4bit 模型目录` \
    --name_or_path Qwen/Qwen3.6-27B \
    `# --out_dir /root/autodl-tmp/MyLLMDataset/QLoRA_model` \
    `# ===== 基础训练参数 =====` \
    `# 27B QLoRA 在 40G A100 上 bs 必须 1，靠 grad_accum 凑 effective bs` \
    --epochs 2 \
    --batch_size 1 \
    `# QLoRA continued pretraining 学习率比 SFT 还要保守，避免冲掉 base 能力` \
    --learning_rate 5e-5 \
    `# bf16：A100 原生支持，不需要 GradScaler` \
    `# ===== 实验跟踪与数据 =====` \
    --num_workers 4 \
    `# --data_path /root/autodl-tmp/MyLLMDataset/processed_dataset/poetry_arrow` \
    `# 启用 SwanLab 取消下一行注释（或命令行追加 --use_swanlab）` \
    --use_swanlab \
    `# ===== 训练优化 =====` \
    `# 27B + bs1 → 用 grad_accum=16 凑 effective batch 16` \
    --gradient_accumulation_steps 16 \
    `# 27B 4bit 不开 GC 必 OOM，强烈建议保留` \
    --gradient_checkpointing \
    --grad_clip 1.0 \
    `# warmup ≈ 第一个 epoch 的 1/3` \
    --warmup_iters 500 \
    --weight_decay 0.1 \
    --betas 0.9 0.95 \
    `# ===== LoRA 超参 =====` \
    `# rank=8, alpha=8 → scale=1，保守。需要更强表达时把 rank 提到 16/32` \
    --lora_rank 8 \
    --lora_alpha 8 \
    --lora_dropout 0.0 \
    `# ===== 日志与保存 =====` \
    `# poetry 9.5M tokens ~290 optimizer steps/epoch，log 频繁点能尽早看 loss 趋势` \
    --log_interval 50 \
    `# save_interval 单位是 per-batch step（不是 optimizer step），2000 step ≈ 半个 epoch` \
    --save_interval 2000 \
    `# snapshot 默认 10**12 = 禁用，要打开就改成合理值如 5000` \
    `# --snapshot_interval 5000` \
    `# ===== 断点续训（默认全新训练；续训时取消下一行并改路径） =====` \
    `# --resume /root/autodl-tmp/MyLLMDataset/QLoRA_model/qlora_epoch1_step2000.pt` \
    "$@"
