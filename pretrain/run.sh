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
# 单机多卡端口
MASTER_PORT="${MASTER_PORT:-29500}"

# NCCL / tokenizers 常用环境变量
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
# export NCCL_DEBUG=INFO
# export NCCL_P2P_DISABLE=1
# export NCCL_IB_DISABLE=1

echo "============================================================"
echo "PROJECT_ROOT       = ${PROJECT_ROOT}"
echo "CUDA_VISIBLE_DEVICES = ${CUDA_VISIBLE_DEVICES}"
echo "NPROC_PER_NODE     = ${NPROC_PER_NODE}"
echo "MASTER_PORT        = ${MASTER_PORT}"
echo "============================================================"

# ===== 训练参数（按需修改）=====
# 任何额外参数都可以从命令行追加，会原样透传给 main.py
# 例如：bash pretrain/run.sh --epochs 2 --batch_size 16 --use_swanlab
torchrun \
    --standalone \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --master_port="${MASTER_PORT}" \
    -m pretrain.main \
    `# ===== 基础训练参数 =====` \
    `# --out_dir /root/autodl-tmp/MyLLMDataset/base_model` \
    --epochs 1 \
    --batch_size 48 \
    --learning_rate 3e-4 \
    --dtype bfloat16 \
    `# ===== 实验跟踪与数据 =====` \
    --num_workers 4 \
    `# --data_path /root/autodl-tmp/MyLLMDataset/processed_dataset/seq_monkey.jsonl` \
    `# 启用 SwanLab 取消下一行注释（或命令行追加 --use_swanlab）` \
    --use_swanlab \
    `# ===== 训练优化 =====` \
    --gradient_accumulation_steps 2 \
    --gradient_checkpointing \
    --grad_clip 1.0 \
    --warmup_iters 750 \
    --weight_decay 0.1 \
    --betas 0.9 0.95 \
    `# ===== 日志与保存 =====` \
    --log_interval 20 \
    --save_interval 500 \
    --snapshot_interval 2000 \
    `# ===== 模型结构（MyModelConfig） =====` \
    --dim 1024 \
    --n_layers 22 \
    --n_heads 8 \
    --n_kv_heads 2 \
    --multiple_of 64 \
    --max_seq_len 1024 \
    --norm_eps 1e-5 \
    --model_dropout 0.0 \
    --flash_attention \
    `# hidden_dim 不传则按 2/3*4*dim 自动计算；如需手动指定取消下一行注释` \
    --hidden_dim 3072 \
    `# ===== MoE（默认关闭，启用时取消注释 --use_moe） =====` \
    `# --use_moe` \
    --n_experts 8 \
    --moe_top_k 3 \
    --router_aux_loss_coef 1e-2 \
    `# ===== 断点续训（默认从头训练；续训时取消下一行注释并改路径） =====` \
    `# --resume /root/autodl-tmp/MyLLMDataset/base_model/pretrain_param_count350M.pt` \
    "$@"
