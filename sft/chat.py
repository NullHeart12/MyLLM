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
                       default=os.path.join(PROJECT_ROOT, "sft_model", "hf_model"),
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

    SYSTEM_PROMPT = (
        "你是 MyLLM，一个由用户从零训练的中文人工智能助手。"
        "请用清晰、准确、有条理的中文回答用户问题；"
        "在不确定时如实说明，不要编造事实；"
        "涉及代码、数学或推理时分步给出过程，最后给出明确结论。"
    )

    print("=" * 60)
    print("MyLLM 多轮对话")
    print("命令：/clear 清空历史，/quit 退出，/sys 改 system prompt")
    print("=" * 60)
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]            
    while True:
        try:
            user_input = input("\n👤 你：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 再见")
            break        
        
        if not user_input:
            continue
        if user_input in ("/quit", "/q", "/exit"):
            print("👋 再见")
            break
        if user_input == "/clear":
            messages = messages[:1]
            print("✅ 历史已清空")
            continue
        if user_input.startswith("/sys:"):
            messages[0]["content"] = user_input[5:]
            messages = messages[:1]
            print(f"✅ system prompt 已改：{messages[0]['content']}")
            continue    
        
        messages.append({"role": "user", "content": user_input})
        
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    
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
                                                    top_k=50)[0, :]
            else:
                generation_ids = my_model.generate(input_ids=tokenized_prompt,
                                                attention_mask=attention_mask,
                                                max_new_tokens=400,
                                                use_cache=False,
                                                do_sample=True,
                                                temperature=0.7,
                                                top_k=50,
                                                pad_token_id=tokenizer.pad_token_id,
                                                eos_token_id=tokenizer.eos_token_id)[
                                                    0, tokenized_prompt.shape[1]:
                                                ]
        
        reply = tokenizer.decode(generation_ids, skip_special_tokens=True)
        messages.append({"role": "assistant", "content": reply})
        print(f"\n🤖 助手：{reply}")
