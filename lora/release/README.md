# Qwen3-8B 古诗词 QLoRA

基于 [Qwen/Qwen3-8B](https://modelscope.cn/models/Qwen/Qwen3-8B) 用 **qlora** 在古诗词数据上做 continued pretraining 得到的 LoRA adapter。

## 训练信息

| 项 | 值 |
|---|---|
| 训练方法 | `qlora` |
| Base 模型 | `Qwen/Qwen3-8B` |
| LoRA rank / alpha | 8 / 8 |
| LoRA target modules | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` |
| 学习率 | 0.0003 |
| 训练数据 | chinese-poetry GitHub(全唐诗 + 全宋词,~30 万首) |
| 数据格式 | `朝代 · 作者《题目》:正文` 自然语言拼接 |
| 评估 PPL | 61.962 |
| 训练日期 | 2026-05-31 |

## 实验追溯

完整 MLflow run id(含 train loss 曲线、超参、上游数据 run id):

- **Training run id**: `b0587bbfb524468dbdbfb69281528297`

## 使用方法

### 1. 加载 base 模型 + adapter

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from modelscope.hub.snapshot_download import snapshot_download

# 下 adapter
adapter_dir = snapshot_download("eihhhaaa/Qwen3-8B-Poetry-QLoRA")
adapter_path = f"{adapter_dir}/lora_adapter.pt"

# 加载 base
base = "Qwen/Qwen3-8B"
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
