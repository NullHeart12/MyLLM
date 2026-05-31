"""
lora/publish_modelscope.py
把训练好的 LoRA adapter 打包成 ModelScope 仓库格式并推送。

输出目录结构(--release_dir 下):
    release/
    ├── lora_adapter.pt        ← 实际的 LoRA 权重(几 MB)
    ├── meta.json              ← 训练超参 + lineage(给程序用)
    └── README.md              ← 介绍 + 使用示例(给人用)

ModelScope 鉴权:
    跑之前先 `modelscope login`(用 ModelScope 网页生成的 token)
    或者设环境变量 `MODELSCOPE_API_TOKEN=...`

用法:
    # 1. 先纯本地组装,看一下产物对不对(不上传)
    python -m lora.publish_modelscope \\
        --adapter model/sweep_out/C-qlora-lr3e-4/qlora_adapter_final.pt \\
        --repo_id eihhhaaa/Qwen3-8B-Poetry-QLoRA \\
        --model_name "Qwen3-8B 古诗词 QLoRA" \\
        --method qlora --rank 8 --alpha 8 --lr 3e-4 \\
        --ppl 61.962 \\
        --mlflow_run_id b0587bbfb524468dbdbfb69281528297 \\
        --no_upload

    # 2. 确认 release/ 内容 OK,再正式上传
    python -m lora.publish_modelscope \\
        --adapter ... --repo_id ... --model_name "..." \\
        --method qlora --rank 8 ... \\
        # 不带 --no_upload 即真上传
"""
import os
import argparse
import json
import shutil
from datetime import datetime

from modelscope.hub.api import HubApi
from modelscope.hub.constants import Licenses, ModelVisibility


README_TEMPLATE = """# {model_name}

基于 [{base_model}](https://modelscope.cn/models/{base_model}) 用 **{method}** 在古诗词数据上做 continued pretraining 得到的 LoRA adapter。

## 训练信息

| 项 | 值 |
|---|---|
| 训练方法 | `{method}` |
| Base 模型 | `{base_model}` |
| LoRA rank / alpha | {rank} / {alpha} |
| LoRA target modules | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` |
| 学习率 | {lr} |
| 训练数据 | chinese-poetry GitHub(全唐诗 + 全宋词,~30 万首) |
| 数据格式 | `朝代 · 作者《题目》:正文` 自然语言拼接 |
| 评估 PPL | {ppl} |
| 训练日期 | {date} |

## 实验追溯

完整 MLflow run id(含 train loss 曲线、超参、上游数据 run id):

- **Training run id**: `{mlflow_run_id}`

## 使用方法

### 1. 加载 base 模型 + adapter

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from modelscope.hub.snapshot_download import snapshot_download

# 下 adapter
adapter_dir = snapshot_download("{repo_id}")
adapter_path = f"{{adapter_dir}}/lora_adapter.pt"

# 加载 base
base = "{base_model}"
model = AutoModelForCausalLM.from_pretrained(
    base,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(base)

# 套 adapter(需要本地有 `model_lora.py`,定义了 LoRALinear / load_lora)
from model_lora import load_lora
load_lora(model, adapter_path)
```

### 2. 生成古诗词

```python
prompt = "唐 · 李白《"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
output = model.generate(
    **inputs,
    max_new_tokens=80,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

### 3. QLoRA 加载(省显存,~5 GB GPU 即可推理)

```python
from transformers import BitsAndBytesConfig
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
model = AutoModelForCausalLM.from_pretrained(
    base, quantization_config=bnb, device_map="auto",
)
# 同上调 load_lora(model, adapter_path)
```

## 训练代码

本项目代码:[github.com/NullHeart12/MyLLM](https://github.com/NullHeart12/MyLLM)(替换成你实际地址)

## License

Apache 2.0
"""


def build_release_dir(adapter_src: str, meta: dict, release_dir: str):
    """组装发布目录:拷 adapter + 写 meta.json + 写 README.md。"""
    os.makedirs(release_dir, exist_ok=True)

    # 1. 拷 adapter
    adapter_dst = os.path.join(release_dir, "lora_adapter.pt")
    shutil.copy(adapter_src, adapter_dst)
    print(f"  ✓ adapter -> {adapter_dst}")

    # 2. 写 meta.json
    meta_path = os.path.join(release_dir, "meta.json")
    with open(meta_path, "w", encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"  ✓ meta.json -> {meta_path}")

    # 3. 写 README.md
    readme_text = README_TEMPLATE.format(**meta)
    readme_path = os.path.join(release_dir, "README.md")
    with open(readme_path, "w", encoding='utf-8') as f:
        f.write(readme_text)
    print(f"  ✓ README.md -> {readme_path}")

    # 总结
    print(f"\n发布目录就绪: {release_dir}")
    for fname in sorted(os.listdir(release_dir)):
        fpath = os.path.join(release_dir, fname)
        size_mb = os.path.getsize(fpath) / 1024 / 1024
        print(f"   {fname}  ({size_mb:.2f} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter",   required=True, help="本地 adapter .pt 路径")
    parser.add_argument("--repo_id",   required=True,
                        help="ModelScope 仓库,格式 username/model-name")
    parser.add_argument("--model_name", required=True,
                        help="人类可读模型名,会作为 chinese_name 和 README 标题")

    parser.add_argument("--base_model",     default="Qwen/Qwen3-8B")
    parser.add_argument("--method",         default="qlora", choices=["lora", "qlora"])
    parser.add_argument("--rank",           type=int,   default=8)
    parser.add_argument("--alpha",          type=int,   default=8)
    parser.add_argument("--lr",             type=float, default=5e-5)
    parser.add_argument("--ppl",            default="N/A",
                        help="评估 PPL(可以是数字字符串或 'N/A')")
    parser.add_argument("--mlflow_run_id",  default="N/A",
                        help="训练 run id(写进 README,方便后续追溯)")

    parser.add_argument("--release_dir", default="./lora/release",
                        help="本地组装目录(上传前可手动检查)")
    parser.add_argument("--no_upload", action="store_true",
                        help="只在本地组装,不上传 ModelScope(用于预览)")
    parser.add_argument("--license", default="apache-2.0",
                        choices=["apache-2.0", "mit", "cc-by-nc-4.0"])
    parser.add_argument("--visibility", default="public", choices=["public", "private"])

    args = parser.parse_args()

    # ===== 1. 准备 meta =====
    meta = {
        "model_name":      args.model_name,
        "base_model":      args.base_model,
        "method":          args.method,
        "rank":            args.rank,
        "alpha":           args.alpha,
        "lr":              args.lr,
        "ppl":             args.ppl,
        "mlflow_run_id":   args.mlflow_run_id,
        "repo_id":         args.repo_id,
        "date":            datetime.now().strftime("%Y-%m-%d"),
    }

    print(f"准备发布: {args.repo_id}")
    print(f"  adapter:  {args.adapter}")
    print(f"  meta:")
    for k, v in meta.items():
        print(f"    {k}: {v}")

    # ===== 2. 组装本地目录 =====
    print(f"\n组装发布目录: {args.release_dir}")
    build_release_dir(args.adapter, meta, args.release_dir)

    # ===== 3. 上传(或跳过) =====
    if args.no_upload:
        print(f"\n[--no_upload] 跳过上传,可手动检查 {args.release_dir} 后再不带 --no_upload 重跑")
        exit(0)

    print(f"\n上传到 ModelScope: {args.repo_id}")
    api = HubApi()

    # 创建仓库(已存在会抛异常,捕获后继续 push)
    visibility = (ModelVisibility.PUBLIC
                  if args.visibility == "public"
                  else ModelVisibility.PRIVATE)
    license_map = {
        "apache-2.0":   Licenses.APACHE_V2,
        "mit":          Licenses.MIT,
    }
    try:
        api.create_model(
            model_id=args.repo_id,
            visibility=visibility,
            license=license_map[args.license],
            chinese_name=args.model_name,
        )
        print(f"  ✓ 仓库创建: {args.repo_id}")
    except Exception as e:
        print(f"  仓库可能已存在,跳过创建步骤: {type(e).__name__}: {e}")

    # 推送目录
    try:
        api.push_model(
            model_id=args.repo_id,
            model_dir=args.release_dir,
        )
        print(f"\n✓ 发布完成!")
        print(f"  https://modelscope.cn/models/{args.repo_id}")
    except Exception as e:
        print(f"\n✗ 上传失败: {type(e).__name__}: {e}")
        print(f"  本地组装目录还在:  {args.release_dir}")
        print(f"  排查后可以单独上传:  modelscope upload {args.repo_id} {args.release_dir}")
        raise
