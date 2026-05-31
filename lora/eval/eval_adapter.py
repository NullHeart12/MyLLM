"""
lora/eval_adapter.py
独立评估脚本。输入 base 模型 + 训练好的 adapter,跑两类评估:
  1. PPL  —— 在 held-out 数据集上算 perplexity
  2. 生成 —— 用一组 prompt 生成续写,做格式合规率检查 + 保存样本

所有指标可选择写回原训练 run(--mlflow_run_id),或者新建一个 eval-only run。

用法:
  # 把指标写回训练 run(推荐 — 在 UI 里看训练 + 评估一体)
  python -m lora.eval.eval_adapter \\
      --base model/Qwen3-8B \\
      --adapter model/sweep_out/A-method-qlora/qlora_adapter_final.pt \\
      --eval_data processed_dataset/chinese_poetry_eval_arrow \\
      --prompts_file lora/eval_prompts.txt \\
      --quantize \\
      --max_eval_samples 100 \\
      --max_new_tokens 150 \\
      --mlflow_run_id <从 MLflow UI 复制>

  # 没 run_id,新建 eval-only run
  python -m lora.eval.eval_adapter \\
      --base model/Qwen3-8B \\
      --adapter model/sweep_out/A-method-qlora/qlora_adapter_final.pt \\
      --eval_data processed_dataset/chinese_poetry_arrow \\
      --quantize \\
      --max_eval_samples 100
"""
import os
import argparse
import math
import json
import re
import random

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Subset
import mlflow

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from ..model_lora import load_lora    # 文件挪到 lora/eval/ 后,model_lora 在父包 lora 里
from deal_dataset.dataset import PretrainDataset


def compute_ppl(model, dataset, device, batch_size=1, max_samples=None):
    """在 dataset 上算 perplexity。max_samples=None 则用全部样本。"""
    model.eval()
    
    # 评估时随机抽样,避免数据顺序导致偏差(唐诗在前宋词在后等)
    # 同一 seed 保证不同 run 评的是同一批样本,对比公平
    if max_samples is not None and max_samples < len(dataset):
        rng = random.Random(42)
        indices = rng.sample(range(len(dataset)), max_samples)
        dataset = Subset(dataset, indices)
    loader = DataLoader(dataset, batch_size=batch_size)

    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if max_samples is not None and i >= max_samples:
                break
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            out = model(input_ids=input_ids, labels=labels, use_cache=False)

            n = (labels != -100).sum().item()
            # 注意:HF model 返回的 loss 是 batch 内所有有效 token 的平均
            total_loss += out.loss.item() * n
            total_tokens += n

    avg_loss = total_loss / max(1, total_tokens)
    ppl = math.exp(avg_loss)
    return ppl, avg_loss, total_tokens


def generate_samples(model, tokenizer, prompts, device,
                     max_new_tokens=80, temperature=0.7, top_p=0.9):
    """对每个 prompt 跑一次生成,返回 [(prompt, full_text), ...]"""
    model.eval()
    results = []
    for prompt in prompts:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            out = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        text = tokenizer.decode(out[0], skip_special_tokens=True)
        results.append((prompt, text))
    return results


def evaluate_format(generations):
    """简单的格律合规检查:有几句、是不是 5/7 言。
    只是粗略指标 —— 古诗严格平仄检查需要专门工具,这里只看基本结构。
    """
    n_total = len(generations)
    if n_total == 0:
        return {}

    n_4lines = 0      # 至少 4 句
    n_5char = 0       # 5 言句数
    n_7char = 0       # 7 言句数

    for _, text in generations:
        # 去掉 metadata 头(":" 之后才是正文)
        body = text.split(":", 1)[-1] if ":" in text else text
        # 按句号切句子
        half_lines = [l.strip() for l in re.split(r'[。,,!?、]', body) if l.strip()]
        if len(half_lines) >= 4:
            n_4lines += 1
        for line in half_lines[:8]:
            # 每行去标点后的字数
            chars = "".join(c for c in line if c not in "。,,!?、 \n")
            if len(chars) == 5:
                n_5char += 1
            elif len(chars) == 7:
                n_7char += 1

    return {
        "format_4lines_rate": n_4lines / n_total,
        "format_5char_lines": n_5char,
        "format_7char_lines": n_7char,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base",         required=True, help="base 模型路径或 HF Hub 名")
    parser.add_argument("--adapter",      required=True, help="LoRA adapter .pt 路径")
    parser.add_argument("--eval_data",    required=True, help="held-out arrow 目录")
    parser.add_argument("--prompts_file", default=None,  
                        help="生成测试 prompt 文件(一行一条);不传则跳过生成评估")

    parser.add_argument("--quantize", action="store_true",
                        help="按 QLoRA 方式加载 base(4bit NF4 + bf16 compute);留空则 bf16 加载")

    parser.add_argument("--max_eval_samples", type=int, default=200,
                        help="评估 PPL 用多少 batch(单 batch=1 sample),None 用全部;评估慢就调小")
    parser.add_argument("--max_new_tokens", type=int, default=80, help="生成最大新 token 数")
    parser.add_argument("--temperature",    type=float, default=0.7)
    parser.add_argument("--top_p",          type=float, default=0.9)

    parser.add_argument("--mlflow_run_id",     default=None,
                        help="把指标写回这个训练 run(推荐:UI 里训练+评估一体)")
    parser.add_argument("--mlflow_experiment", default="MyLLM-LoRA-Eval",
                        help="如果没传 run_id,在这个 experiment 里新建 eval-only run")

    parser.add_argument("--device",  default="cuda:0")
    parser.add_argument("--out_dir", default="./lora/eval_out",
                        help="生成样本和指标 json 的本地保存目录")

    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # ===== 1. 加载 base 模型 =====
    print(f"加载 base: {args.base}  (quantize={args.quantize})")
    if args.quantize:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.base,
            quantization_config=bnb_config,
            device_map={"": 0},
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.base,
            torch_dtype=torch.bfloat16,
            device_map={"": 0},
        )
    model.config.use_cache = True   # 评估时打开 KV cache,生成更快

    tokenizer = AutoTokenizer.from_pretrained(args.base)

    print(f"加载 adapter: {args.adapter}")
    load_lora(model, args.adapter)
    model.eval()

    # ===== 2. PPL 评估 =====
    print(f"PPL 评估(数据: {args.eval_data}, max_samples={args.max_eval_samples})")
    eval_ds = PretrainDataset(args.eval_data)
    ppl, avg_loss, n_tokens = compute_ppl(
        model, eval_ds, args.device,
        batch_size=8,
        max_samples=args.max_eval_samples,
    )
    print(f"  PPL: {ppl:.3f}  avg_loss: {avg_loss:.4f}  n_tokens: {n_tokens}")

    # ===== 3. 生成评估(可选) =====
    generations = []
    format_stats = {}
    gen_path = None
    if args.prompts_file and os.path.exists(args.prompts_file):
        with open(args.prompts_file, encoding='utf-8') as f:
            prompts = [line.strip() for line in f if line.strip() and not line.startswith("#")]

        if prompts:
            print(f"生成评估({len(prompts)} 条 prompt)")
            generations = generate_samples(
                model, tokenizer, prompts, args.device,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            format_stats = evaluate_format(generations)

            # 保存生成样本到本地
            gen_path = os.path.join(args.out_dir, "generations.txt")
            with open(gen_path, "w", encoding='utf-8') as f:
                f.write(f"# Adapter: {args.adapter}\n")
                f.write(f"# PPL:     {ppl:.3f}\n")
                f.write(f"# Format:  {format_stats}\n\n")
                for p, t in generations:
                    f.write(f"PROMPT: {p}\n")
                    f.write(f"GEN:    {t}\n")
                    f.write("-" * 60 + "\n")
            print(f"  生成样本保存到 {gen_path}")

            # 打印前 3 条让用户即时看到效果
            for p, t in generations[:3]:
                preview = t[:120].replace("\n", " ")
                print(f"  PROMPT: {p}")
                print(f"  GEN:    {preview}...")

    # ===== 4. 写回 MLflow =====
    metrics = {
        "eval/ppl":            ppl,
        "eval/avg_loss":       avg_loss,
        "eval/n_tokens":       float(n_tokens),
        **{f"eval/{k}": float(v) for k, v in format_stats.items()},
    }
    # 同时存一份 json 到本地(MLflow 挂掉时兜底)
    metrics_path = os.path.join(args.out_dir, "metrics.json")
    with open(metrics_path, "w", encoding='utf-8') as f:
        json.dump({"adapter": args.adapter, "metrics": metrics}, f, indent=2)
    print(f"指标 json 保存到 {metrics_path}")

    try:
        if args.mlflow_run_id:
            # 写回训练 run
            with mlflow.start_run(run_id=args.mlflow_run_id):
                mlflow.log_metrics(metrics)
                if gen_path:
                    mlflow.log_artifact(gen_path, artifact_path="eval")
                mlflow.log_artifact(metrics_path, artifact_path="eval")
            print(f"✓ 指标写回训练 run: {args.mlflow_run_id}")
        else:
            # 新建 eval-only run
            mlflow.set_experiment(args.mlflow_experiment)
            run_name = f"eval-{os.path.basename(args.adapter).replace('.pt','')}"
            with mlflow.start_run(run_name=run_name):
                mlflow.set_tag("type", "eval_only")
                mlflow.set_tag("adapter", args.adapter)
                mlflow.set_tag("base", args.base)
                mlflow.log_metrics(metrics)
                if gen_path:
                    mlflow.log_artifact(gen_path, artifact_path="eval")
                mlflow.log_artifact(metrics_path, artifact_path="eval")
            print(f"✓ 新建 eval-only run 写入 experiment: {args.mlflow_experiment}")
    except Exception as e:
        print(f"⚠️ MLflow 写入失败(但本地 json 已保存): {e}")

    print("\n===== 评估完成 =====")
    print(f"PPL:       {ppl:.3f}")
    if format_stats:
        print(f"格式合规:   {format_stats}")
    print(f"本地输出:  {args.out_dir}")
