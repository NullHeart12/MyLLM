import os
import argparse

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

from model import Transformer, MyModelConfig
from train_utils import count_parameters

if __name__ == "__main__":
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # PROJECT_ROOT = "/root/autodl-tmp/MyLLMDataset"
    
    parse = argparse.ArgumentParser(description="使用预训练模型进行文本生成")
    parse.add_argument("--model_path", type=str,
                       default=os.path.join(PROJECT_ROOT, "base_model", "hf_model_291M"),
                    #    default=os.path.join(PROJECT_ROOT, "base_model", "hf_model_336M"),
                       help="预训练模型路径（HF 格式目录，含 config.json + model.safetensors + tokenizer 文件）")
    parse.add_argument("--use_auto", action="store_true", help="是否使用 AutoModelForCausalLM 加载模型")
    parse.add_argument("--use_my_gen", action="store_true", help="是否使用自定义生成函数")
    
    args = parse.parse_args()
        
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
        
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        assert False, "当前环境没有可用的 GPU，无法进行生成测试。请在支持 CUDA 的环境中运行此脚本。"

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    if args.use_auto:
        AutoConfig.register("my_model", MyModelConfig)
        AutoModelForCausalLM.register(MyModelConfig, Transformer)
        my_model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16)
    else:
        my_model = Transformer.from_pretrained(args.model_path, torch_dtype=torch.bfloat16)
    my_model.to(device)

    num_params = count_parameters(my_model)
    print(f"模型参数量: {num_params / 1e6:.3f}M ({num_params:,})")

    # 打印模型配置
    cfg = my_model.config
    print("=" * 60)
    print("模型配置:")
    for k, v in sorted(cfg.to_dict().items()):
        print(f"  {k:30s} = {v}")
    print("=" * 60)

    prompt = [
        # "合肥工业大学是",
        # "中国科学院是",
        # "中国科学院计算技术研究所位于北京市中关村，该研究所为中国",
        # "人工智能是计算机科学的一个分支，它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出",
        # "卧槽，尼玛，这是",
        # "I am very happy to be",
        # "When I was",
        # "Oh motherfuker,shit bitch",
        # "Based on the comparative expression of the single-copy",
        # "Large language models (LLMs) are a type of artificial intelligence (AI) trained on vast",
        # "Python 是一种高级编程语言，以其简洁的语法和强大的生态系统而闻名。It is widely used in data science, "
    ]
    
    tokenizer.padding_side = "left"
    tokenized = tokenizer(prompt,
                          add_special_tokens=True, 
                          return_tensors="pt",
                          padding=True,
                          padding_side="left",
                          return_attention_mask=True).to(device)
    
    tokenized_prompt = tokenized['input_ids']
    attention_mask = tokenized['attention_mask']
    
    my_model.eval()
    with torch.inference_mode():
        if args.use_my_gen:
            generation_ids = my_model.my_generate(input_ids=tokenized_prompt,
                                                  key_padding_mask=attention_mask,
                                                  max_new_tokens=400,
                                                  temperature=0.7,
                                                  top_k=50)[:, tokenized_prompt.shape[1]:]
        else:
            generation_ids = my_model.generate(input_ids=tokenized_prompt,
                                               attention_mask=attention_mask,
                                               max_new_tokens=400,
                                               use_cache=False,
                                               do_sample=True,
                                               temperature=0.7,
                                               top_k=50,
                                               pad_token_id=tokenizer.pad_token_id,
                                               eos_token_id=tokenizer.eos_token_id)
    
    generation = tokenizer.batch_decode(generation_ids, skip_special_tokens=True)

    gen_mode = "my_generate" if args.use_my_gen else "hf.generate"
    print("\n" + "=" * 80)
    print(f"生成模式: {gen_mode}  |  样本数: {len(prompt)}  |  device: {device}")
    print("=" * 80)
    for i, (p, g) in enumerate(zip(prompt, generation), start=1):
        print(f"\n[{i}/{len(prompt)}] ---------- PROMPT ----------")
        print(p)
        print(f"[{i}/{len(prompt)}] ---------- OUTPUT ----------")
        print(g)
        print("-" * 80)
