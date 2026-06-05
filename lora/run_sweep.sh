#!/usr/bin/env bash
# lora/run_sweep.sh
# 跑一组对比实验,所有 run 写入同一个 MLflow experiment,UI 里勾选多个 → Compare 即可。
#
# 三组对比:
#   A. LoRA vs QLoRA(rank=8, lr=5e-5 固定,只换方法)         → 2 个 run
#   B. QLoRA rank 扫描(lr=5e-5 固定,rank ∈ {4,8,16,32})     → 4 个 run
#   C. QLoRA lr 扫描(rank=8 固定,lr ∈ {1e-5,5e-5,1e-4,3e-4}) → 4 个 run
# 总共 10 个 run。Qwen3-8B QLoRA 单 epoch 大约 5-15 min/run,跑完约 1-2 小时。
#
# 用法:
#   bash lora/run_sweep.sh                  # 默认全跑
#   ENTRY_SCRIPT=lora/run_lora.sh bash ...  # 换 entry 脚本(如果以后拆分了)
#   MLFLOW_EXPERIMENT=xxx bash ...          # 换实验名
#
# 失败处理:某个 run 崩了不会中断其他 run,最后会列出失败列表。

set -uo pipefail   # 故意不用 -e,某个 run 失败继续下一个

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# ===== 可配置 =====
ENTRY_SCRIPT="${ENTRY_SCRIPT:-lora/run_lora.sh}"
MLFLOW_EXPERIMENT="${MLFLOW_EXPERIMENT:-MyLLM-LoRA-Sweep}"
SWEEP_OUT_BASE="${SWEEP_OUT_BASE:-${PROJECT_ROOT}/model/sweep_out}"

# ===== 所有 run 共用的固定参数(显式列出,不依赖 argparse 默认) =====
# 原则:sweep 只想让 method / rank / lr 变化,其他维度全锁死。
# 显式写出来的好处:以后 argparse 默认变了,sweep 行为不受影响,可复现。
COMMON=(
    # ----- 模型与数据(对比实验的基石,必须固定) -----
    --name_or_path "${PROJECT_ROOT}/model/Qwen3-8B"
    --data_path    "${PROJECT_ROOT}/processed_dataset/chinese_poetry_arrow"

    # ----- 训练循环 -----
    --epochs 1                           # sweep 单 epoch 足够看趋势
    --batch_size 8
    --gradient_accumulation_steps 2      # effective batch = 1 × 16 = 16
    --gradient_checkpointing             # 必开,否则 OOM
    --warmup_iters 20                    # 单 epoch 不需要长 warmup

    # ----- 优化器(锁住,只让 lr 在 C 组里变) -----
    --weight_decay 0.1
    --grad_clip 1.0
    --betas 0.9 0.95

    # ----- LoRA(锁住,只让 rank/method 在 A/B 组里变;C 组也保持 dropout=0) -----
    --lora_dropout 0.0

    # ----- 跟踪 & 保存 -----
    --use_swanlab
    --use_mlflow
    --mlflow_experiment "${MLFLOW_EXPERIMENT}"
    --log_interval 10
    --save_interval 10000000              # sweep 不需要中间 ckpt,设极大值禁用
    --num_workers 2                      # 单卡用 2 个 worker 就够;数据量大可调高
)

# 跟踪状态
OK_RUNS=()
FAILED_RUNS=()
TOTAL_START=$(date +%s)

# ===== 单 run 包装函数 =====
run_one() {
    local tag="$1"; shift
    local out_dir="${SWEEP_OUT_BASE}/${tag}"

    echo ""
    echo "═══════════════════════════════════════════════════════════"
    echo "▶ Sweep run: ${tag}"
    echo "  Extra args: $*"
    echo "  Out dir:    ${out_dir}"
    echo "═══════════════════════════════════════════════════════════"

    local start_ts=$(date +%s)

    # 注意参数顺序:COMMON 在前,per-run "$@" 在后(argparse 后出现的覆盖前面的,
    # 这样 sweep 调用方传 --lora_rank 16 可以覆盖 COMMON 里的默认值)
    bash "${ENTRY_SCRIPT}" \
        --out_dir "${out_dir}" \
        --mlflow_run_name "${tag}" \
        "${COMMON[@]}" \
        "$@"
    local rc=$?

    local elapsed=$(($(date +%s) - start_ts))
    local mins=$((elapsed / 60))
    local secs=$((elapsed % 60))

    if [[ $rc -ne 0 ]]; then
        echo "✗ ${tag} FAILED (exit=$rc, 耗时 ${mins}m${secs}s) —— 继续下一个"
        FAILED_RUNS+=("${tag}")
    else
        echo "✓ ${tag} OK (耗时 ${mins}m${secs}s)"
        OK_RUNS+=("${tag}")
    fi
}

echo "════════════════════════════════════════"
echo " Sweep 开始"
echo "   Entry script:       ${ENTRY_SCRIPT}"
echo "   MLflow experiment:  ${MLFLOW_EXPERIMENT}"
echo "   Adapter 输出根目录:  ${SWEEP_OUT_BASE}"
echo "════════════════════════════════════════"

mkdir -p "${SWEEP_OUT_BASE}"

# ===== A. LoRA vs QLoRA 方法对比 =====
# 固定其他超参,只换方法。看精度/速度/显存差异。
for method in qlora lora; do
    run_one "A-method-${method}" \
        --lora_method "$method" \
        --lora_rank 8 --lora_alpha 8 \
        --learning_rate 5e-5
done

# ===== B. QLoRA rank 扫描 =====
# 看 rank 对效果/参数量的影响。alpha 跟 rank 保持 1:1(scale=1)。
for rank in 4 8 16 32; do
    run_one "B-qlora-rank${rank}" \
        --lora_method qlora \
        --lora_rank "$rank" --lora_alpha "$rank" \
        --learning_rate 5e-5
done

# ===== C. QLoRA learning rate 扫描 =====
# 看 lr 对收敛速度/最终 loss 的影响。
for lr in 1e-5 5e-5 1e-4 3e-4; do
    run_one "C-qlora-lr${lr}" \
        --lora_method qlora \
        --lora_rank 8 --lora_alpha 8 \
        --learning_rate "$lr"
done

# ===== 总结 =====
TOTAL_ELAPSED=$(($(date +%s) - TOTAL_START))
TOTAL_MINS=$((TOTAL_ELAPSED / 60))

echo ""
echo "════════════════════════════════════════════════════════"
echo " ✓ Sweep 全部跑完 (总耗时 ${TOTAL_MINS} 分)"
echo "════════════════════════════════════════════════════════"
echo " 成功 ${#OK_RUNS[@]} 个:"
for r in "${OK_RUNS[@]}"; do echo "   ✓ $r"; done
if [[ ${#FAILED_RUNS[@]} -gt 0 ]]; then
    echo ""
    echo " 失败 ${#FAILED_RUNS[@]} 个:"
    for r in "${FAILED_RUNS[@]}"; do echo "   ✗ $r"; done
fi
echo ""
echo " 在 MLflow UI 里查看 (实验: ${MLFLOW_EXPERIMENT}):"
echo "   mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000 \\"
echo "       --host 0.0.0.0 --allowed-hosts '*' --cors-allowed-origins '*'"
echo "════════════════════════════════════════════════════════"

# 如果有失败,exit code 非 0(方便上层 CI 检测)
[[ ${#FAILED_RUNS[@]} -eq 0 ]]
