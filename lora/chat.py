# # bf16 加载(显存约 16GB)
# python -m lora.chat

# # 4bit 加载(显存约 6GB,适合显存紧张时)
# python -m lora.chat --quantize

# # 用其他 adapter
# python -m lora.chat --adapter model/sweep_out/B-qlora-rank32/qlora_adapter_final.pt --quantize


import os
import argparse

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from .model_lora import load_lora    # 文件挪到 lora/eval/ 后,model_lora 在父包 lora 里

class QwenChatbot:
    def __init__(self, args: argparse.Namespace = None):
        # ===== 1. 加载 base 模型 =====
        print(f"加载 base: {args.base}  (quantize={args.quantize})")
        if args.quantize:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                args.base,
                quantization_config=bnb_config,
                device_map={"": 0},
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                args.base,
                torch_dtype=torch.bfloat16,
                device_map={"": 0},
            )
        self.model.config.use_cache = True   # 评估时打开 KV cache,生成更快

        self.tokenizer = AutoTokenizer.from_pretrained(args.base)

        # ===== 2. 是否加载 adapter =====
        # --no_adapter 时跳过 LoRA 注入,纯 base 模型对话(用于对比 base vs fine-tuned)
        if args.no_adapter:
            print("跳过 adapter 加载,使用纯 base 模型")
            self.adapter_loaded = False
            self.adapter_path = None
        else:
            print(f"加载 adapter: {args.adapter}")
            load_lora(self.model, args.adapter)
            self.adapter_loaded = True
            self.adapter_path = args.adapter

        self.model.eval()
        self.history = []

        # Qwen3 thinking mode (默认开启,生成 <think>...</think> 推理段)
        # --no_think 启动时关闭;运行时可用 /think /no_think 切换
        self.enable_thinking = not args.no_think

    @torch.inference_mode()
    def generate_response(self, user_input):
        messages = self.history + [{"role": "user", "content": user_input}]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )

        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        response_ids = self.model.generate(**inputs, max_new_tokens=32768)[0][len(inputs.input_ids[0]):].tolist()
        response = self.tokenizer.decode(response_ids, skip_special_tokens=True)

        # Update history
        self.history.append({"role": "user", "content": user_input})
        self.history.append({"role": "assistant", "content": response})

        return response

    def clear_history(self):
        """清空多轮对话上下文,回到首次启动状态。"""
        n = len(self.history)
        self.history = []
        return n  # 返回被清掉的消息条数,方便 caller 打印


if __name__ == "__main__":
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--base",         
                        default=os.path.join(PROJECT_ROOT, "model", "Qwen3-8B"),
                        help="base 模型路径或 HF Hub 名")
    parser.add_argument("--adapter",
                        default=os.path.join(
                            PROJECT_ROOT,
                            "model",
                            "LoRA_out",
                            "qlora_adapter_final.pt"
                        ),
                        help="LoRA adapter .pt 路径(--no_adapter 时忽略)")
    parser.add_argument("--no_adapter", action="store_true",
                        help="不加载 adapter,纯 base 模型对话(用于对比 base vs fine-tuned 效果)")
    parser.add_argument("--no_think", action="store_true",
                        help="关闭 Qwen3 thinking mode(默认开启,会生成 <think>...</think> 推理段)")

    parser.add_argument("--quantize", action="store_true",
                        help="按 QLoRA 方式加载 base(4bit NF4 + bf16 compute);留空则 bf16 加载")

    parser.add_argument("--max_new_tokens", type=int, default=80, help="生成最大新 token 数")
    parser.add_argument("--temperature",    type=float, default=0.7)
    parser.add_argument("--top_p",          type=float, default=0.9)

    parser.add_argument("--device",  default="cuda:0")

    args = parser.parse_args()

    chatbot = QwenChatbot(args=args)

    HELP_TEXT = (
        "可用命令:\n"
        "  /help                显示此帮助\n"
        "  /clear               清空对话历史,重置上下文\n"
        "  /history             查看当前已有的历史轮数\n"
        "  /think | /no_think   开启 / 关闭 Qwen3 thinking 模式\n"
        "  /status              查看当前 adapter / thinking 状态\n"
        "  /quit | /q | /exit   退出聊天"
    )
    def _think_label(on: bool) -> str:
        return "🧠 ON" if on else "💤 OFF"

    print("\n" + "=" * 60)
    if chatbot.adapter_loaded:
        print(f"💬 Qwen3 chatbot 已就绪 · 模式: 🎯 base + adapter")
        print(f"   adapter:  {chatbot.adapter_path}")
    else:
        print(f"💬 Qwen3 chatbot 已就绪 · 模式: 🅱 base only (无 adapter)")
    print(f"   thinking: {_think_label(chatbot.enable_thinking)}")
    print(HELP_TEXT)
    print("=" * 60)

    while True:
        try:
            user_input = input("\n👤 你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 再见")
            break

        if not user_input:
            continue
        if user_input in ("/quit", "/q", "/exit"):
            print("👋 再见")
            break
        if user_input == "/help":
            print(HELP_TEXT)
            continue
        if user_input == "/clear":
            n = chatbot.clear_history()
            print(f"✅ 历史已清空(共清掉 {n} 条消息)")
            continue
        if user_input == "/history":
            n_turns = len(chatbot.history) // 2
            print(f"📚 当前历史: {n_turns} 轮({len(chatbot.history)} 条消息)")
            continue
        if user_input == "/think":
            chatbot.enable_thinking = True
            print(f"🧠 thinking 已开启")
            continue
        if user_input == "/no_think":
            chatbot.enable_thinking = False
            print(f"💤 thinking 已关闭")
            continue
        if user_input == "/status":
            adapter_str = (f"🎯 base + adapter ({chatbot.adapter_path})"
                           if chatbot.adapter_loaded else "🅱 base only")
            print(f"📊 状态:")
            print(f"   模式:     {adapter_str}")
            print(f"   thinking: {_think_label(chatbot.enable_thinking)}")
            print(f"   历史:     {len(chatbot.history)//2} 轮")
            continue

        # === 正常对话:调模型生成 ===
        try:
            response = chatbot.generate_response(user_input)
        except KeyboardInterrupt:
            print("\n⏸  生成被中断")
            continue
        except Exception as e:
            print(f"\n✗ 生成失败: {type(e).__name__}: {e}")
            continue

        print(f"\n🤖 助手: {response}")
