#!/usr/bin/env bash
# lora/eval/run_eval.sh
# 批量评估多个 adapter,自动写回 MLflow。
#
# === 用法 ===
# 1. 编辑下面的 RUNS 数组,填入你要评估的所有 adapter 和对应训练 run id
# 2. bash lora/eval/run_eval.sh
# 3. 某个 run 评估失败不影响后续,最后会列出失败列表

set -uo pipefail   # 不用 -e,某个 run 崩了继续下一个

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

# ====================================================================
# ===== 编辑这里:每行一条 "adapter_path|mlflow_run_id"          =====
# =====   adapter_path 必填; mlflow_run_id 留空则新建 eval-only run =====
# =====   填完保存,跑 bash lora/eval/run_eval.sh                  =====
# ====================================================================
RUNS=(
    # === A 组:LoRA vs QLoRA ===
    "model/sweep_out/A-method-qlora/qlora_adapter_final.pt|b469185bd67448ecb249c3e2ec3562e9"
    "model/sweep_out/A-method-lora/lora_adapter_final.pt|939e023fef644a36a400d5be477351ed"

    # === B 组:rank 扫描 ===
    "model/sweep_out/B-qlora-rank4/qlora_adapter_final.pt|8d8247ce753442d4859d00f891b2c013"
    "model/sweep_out/B-qlora-rank8/qlora_adapter_final.pt|fe7cce83c5cb4ebab77d9bcda928d4e5"
    "model/sweep_out/B-qlora-rank16/qlora_adapter_final.pt|96f34864779a495dac47a9e77f90cbbb"
    "model/sweep_out/B-qlora-rank32/qlora_adapter_final.pt|72f08f5dcc0f464788cc9476edc80064"

    # === C 组:learning rate 扫描 ===
    "model/sweep_out/C-qlora-lr1e-5/qlora_adapter_final.pt|d0b41653fa7c4e42b526ead13c719c9b"
    "model/sweep_out/C-qlora-lr5e-5/qlora_adapter_final.pt|2ff15fda71ce4124b6ce29058c56635e"
    "model/sweep_out/C-qlora-lr1e-4/qlora_adapter_final.pt|ec639e61732b4a949799e037f8926452"
    "model/sweep_out/C-qlora-lr3e-4/qlora_adapter_final.pt|b0587bbfb524468dbdbfb69281528297"
)

# ====================================================================
# ===== 其他参数(已配好,按需调) =====
# ====================================================================
BASE_MODEL="${PROJECT_ROOT}/model/Qwen3-8B"
EVAL_DATA="${PROJECT_ROOT}/processed_dataset/chinese_poetry_arrow"
PROMPTS_FILE="${SCRIPT_DIR}/eval_prompts.txt"

MAX_EVAL_SAMPLES=100       # PPL 用多少 batch,越大越准但越慢;100 ~ 2 min/run
MAX_NEW_TOKENS=150
TEMPERATURE=0.7
TOP_P=0.9

OUT_DIR_BASE="${SCRIPT_DIR}/eval_out"
MLFLOW_EXPERIMENT="MyLLM-LoRA-Eval"

# ====================================================================
# ===== 循环 + 状态跟踪(下面一般不用动) =====
# ====================================================================

OK_RUNS=()
FAILED_RUNS=()
SKIPPED_RUNS=()
TOTAL_START=$(date +%s)

echo "════════════════════════════════════════════"
echo " 批量 Eval 开始"
echo "   待评估:        ${#RUNS[@]} 个 adapter"
echo "   base:           ${BASE_MODEL}"
echo "   eval_data:      ${EVAL_DATA}"
echo "   eval_samples:   ${MAX_EVAL_SAMPLES}"
echo "════════════════════════════════════════════"

for entry in "${RUNS[@]}"; do
    # 解析 "adapter|run_id" 两段,| 之前是 adapter,之后是 run_id(可空)
    ADAPTER="${entry%%|*}"
    MLFLOW_RUN_ID="${entry#*|}"

    # adapter 路径若用相对路径,基于 PROJECT_ROOT
    if [[ "${ADAPTER}" != /* ]]; then
        ADAPTER="${PROJECT_ROOT}/${ADAPTER}"
    fi

    # 从 adapter 路径提取 tag(用作 out_dir 子目录 + 日志显示)
    tag="$(basename "$(dirname "${ADAPTER}")")"
    out_dir="${OUT_DIR_BASE}/${tag}"

    # 根据 adapter 文件名判断训练方法:qlora_adapter_*.pt → 加 --quantize
    # 评估条件必须和训练时一致,否则 PPL 不可横比
    adapter_basename="$(basename "${ADAPTER}")"
    if [[ "${adapter_basename}" == qlora_* ]]; then
        METHOD="qlora"
    else
        METHOD="lora"
    fi

    echo ""
    echo "═══════════════════════════════════════════════════════════"
    echo "▶ Eval [${tag}]  (method=${METHOD})"
    echo "   adapter:   ${ADAPTER}"
    echo "   run_id:    ${MLFLOW_RUN_ID:-<新建 eval-only run>}"
    echo "═══════════════════════════════════════════════════════════"

    # 不存在的 adapter 跳过
    if [[ ! -f "${ADAPTER}" ]]; then
        echo "✗ 跳过:adapter 文件不存在"
        SKIPPED_RUNS+=("${tag}")
        continue
    fi

    # 组装命令
    CMD=(
        python -m lora.eval.eval_adapter
        --base              "${BASE_MODEL}"
        --adapter           "${ADAPTER}"
        --eval_data         "${EVAL_DATA}"
        --prompts_file      "${PROMPTS_FILE}"
        --max_eval_samples  "${MAX_EVAL_SAMPLES}"
        --max_new_tokens    "${MAX_NEW_TOKENS}"
        --temperature       "${TEMPERATURE}"
        --top_p             "${TOP_P}"
        --out_dir           "${out_dir}"
        --mlflow_experiment "${MLFLOW_EXPERIMENT}"
    )
    if [[ "${METHOD}" == "qlora" ]]; then
        CMD+=(--quantize)
    fi
    if [[ -n "${MLFLOW_RUN_ID}" ]]; then
        CMD+=(--mlflow_run_id "${MLFLOW_RUN_ID}")
    fi

    # 计时跑
    start_ts=$(date +%s)
    "${CMD[@]}"
    rc=$?
    elapsed=$(($(date +%s) - start_ts))
    mins=$((elapsed / 60))
    secs=$((elapsed % 60))

    if [[ $rc -ne 0 ]]; then
        echo "✗ ${tag} FAILED (exit=$rc, ${mins}m${secs}s) —— 继续下一个"
        FAILED_RUNS+=("${tag}")
    else
        echo "✓ ${tag} OK (${mins}m${secs}s)"
        OK_RUNS+=("${tag}")
    fi
done

# ====================================================================
# ===== 总结 =====
# ====================================================================
TOTAL_ELAPSED=$(($(date +%s) - TOTAL_START))
TOTAL_MINS=$((TOTAL_ELAPSED / 60))

echo ""
echo "════════════════════════════════════════════════════════"
echo " ✓ 批量 Eval 完成 (总耗时 ${TOTAL_MINS} 分)"
echo "════════════════════════════════════════════════════════"
echo " 成功 ${#OK_RUNS[@]} 个:"
for r in "${OK_RUNS[@]}"; do echo "   ✓ $r"; done
if [[ ${#SKIPPED_RUNS[@]} -gt 0 ]]; then
    echo ""
    echo " 跳过 ${#SKIPPED_RUNS[@]} 个(adapter 文件不存在):"
    for r in "${SKIPPED_RUNS[@]}"; do echo "   - $r"; done
fi
if [[ ${#FAILED_RUNS[@]} -gt 0 ]]; then
    echo ""
    echo " 失败 ${#FAILED_RUNS[@]} 个:"
    for r in "${FAILED_RUNS[@]}"; do echo "   ✗ $r"; done
fi
echo ""
echo " 查看 MLflow UI 里 ${MLFLOW_EXPERIMENT} 实验对比所有 eval"
echo "════════════════════════════════════════════════════════"

[[ ${#FAILED_RUNS[@]} -eq 0 ]]
