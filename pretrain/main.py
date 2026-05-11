import os
import argparse

from model import MyModelConfig, Transformer
from ddp_pretrain import set_ddp, destroy_ddp, logger, load_tokenizer, load_model

import swanlab
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

if __name__ == "__main__":
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    parser = argparse.ArgumentParser(description="using DDP Pretrain MyLLM")
    
    # 基础训练参数
    parser.add_argument("--out_dir", type=str, 
                        default=os.path.join(PROJECT_ROOT, "base_model"), 
                        help="模型输出目录")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="每张卡的批次大小（DDP 约定）。全局有效 batch = batch_size × world_size × accumulation_steps")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="学习率")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型: float16 / bfloat16 / float32")

    # 实验跟踪和数据加载参数
    parser.add_argument("--use_swanlab", action="store_true", help="是否使用SwanLab进行实验跟踪")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载的工作进程数")
    parser.add_argument("--data_path", type=str, 
                        default=os.path.join(PROJECT_ROOT, "processed_dataset", "seq_monkey.jsonl"), 
                        help="训练数据路径")

    # 训练优化参数
    parser.add_argument("--accumulation_steps", type=int, default=8, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--warmup_iters", type=int, default=0, help="学习率预热迭代次数")

    # 日志和保存参数
    parser.add_argument("--log_interval", type=int, default=100, help="日志记录间隔")
    parser.add_argument("--save_interval", type=int, default=1000, help="模型保存间隔")    
    
    args = parser.parse_args()
    
    # DDP 初始化
    local_rank = set_ddp()
    args.local_rank = local_rank
    args.device = f"cuda:{local_rank}"
    args.world_size = dist.get_world_size()
    args.rank = dist.get_rank()
    args.is_main = (args.rank == 0)
    logger(f"ddp 初始化完成，world_size:{args.world_size}，backend=nccl")

    # SwanLab 实验跟踪
    if args.use_swanlab and args.is_main:
        swanlab.init(
            project="MyLLM",
            experiment_name=f"DDP-Pretrain",
            config=args
        )
    
    # 加载 tokenizer 和模型
    torch.manual_seed(42)  # 设置随机种子以确保每个进程加载相同的模型权重
    tokenizer = load_tokenizer(os.path.join(PROJECT_ROOT, "tokenizer_k"))
    
    lm_config = MyModelConfig(
        vocab_size=tokenizer.vocab_size,
        dropout=0.1,
        flash_attention=True,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    
    my_model = load_model(lm_config, args)
    
    # 训练准备
    if args.is_main:
        args.save_dir = os.path.join(args.out_dir)
        os.makedirs(args.save_dir, exist_ok=True)
    dist.barrier()  # 确保所有进程都完成了模型加载和准备
    
    # 根据 args.dtype 设置混合精度上下文
    if args.dtype == "bfloat16":
        amp_type = torch.bfloat16
    else:
        amp_type = torch.float16
    ctx = torch.amp.autocast(device_type="cuda", dtype=amp_type)
    scaler = torch.amp.GradScaler(enabled=(args.dtype == "float16"))
    
    
    