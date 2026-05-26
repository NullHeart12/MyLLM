import os
import argparse
import random
import numpy as np

from model import MyModelConfig, Transformer
from train_utils import (
    set_ddp, destroy_ddp, logger,
    load_tokenizer, load_model,
    build_optimizer, maybe_resume, save_hfckpt,
    epoch_train, count_parameters
)
from deal_dataset.dataset import SFTDataset, SFTCollator

import swanlab
import torch
import torch.distributed as dist
from torch import amp
from torch.utils import data
from transformers import (
    AutoConfig, AutoTokenizer, 
    AutoModelForCausalLM,
)
from transformers.trainer_pt_utils import DistributedLengthGroupedSampler

if __name__=="__main__":
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # PROJECT_ROOT = "/root/autodl-tmp/MyLLMDataset"
    
    parser = argparse.ArgumentParser(description="Train QWen3.6-27B with QLoRA")
    
    # 基础训练参数
    parser.add_argument("--path_or_name", type=str, 
                        default="Qwen/Qwen3.6-27B",
                        help="Qwen3.6-27B 模型路径或名称（HF Hub）")
    parser.add_argument("--out_dir", type=str, 
                        default=os.path.join(PROJECT_ROOT, "QLoRA_model"), 
                        help="模型输出目录")
    
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=32, 
                        help="每张卡的批次大小（DDP 约定）。全局有效 batch = batch_size × world_size × gradient_accumulation_steps")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="学习率")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型: float16 / bfloat16 / float32")

    # 实验跟踪和数据加载参数
    parser.add_argument("--use_swanlab", action="store_true", help="是否使用SwanLab进行实验跟踪")
    parser.add_argument("--num_workers", type=int, default=4, help="数据加载的工作进程数")
    parser.add_argument("--data_path", type=str, 
                        # default=os.path.join(PROJECT_ROOT, "processed_dataset", 
                                            #  "BelleGroup_sft.jsonl"), 
                        default=os.path.join(PROJECT_ROOT, "processed_dataset", 
                                             "BelleGroup_sft_tokenized_arrow"), 
                        help="训练数据路径")

    # 训练优化参数
    parser.add_argument("--gradient_accumulation_steps",
                        type=int, default=2, help="梯度累积步数")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        help="启用梯度检查点：以重算前向为代价，按 TransformerBlock 粒度减少显存占用")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--warmup_iters", type=int, default=750, help="学习率预热迭代次数")
    parser.add_argument("--weight_decay", type=float, default=0.1, 
                        help="AdamW weight decay")
    parser.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.95), 
                        help="AdamW betas")

    # 日志和保存参数
    parser.add_argument("--log_interval", type=int, default=150, help="日志记录间隔")
    parser.add_argument("--save_interval", type=int, default=2000, help="模型保存间隔")
    parser.add_argument("--snapshot_interval", type=int, default=0, help="生成快照间隔")    
    
    # 断点续训
    parser.add_argument("--resume", type=str, default=None,
                        # default=os.path.join(PROJECT_ROOT, "sft_model", "pretrain_param_count82.595M.pt"),
                        help="从指定 checkpoint 路径恢复训练")

    args = parser.parse_args()
    
    # 性能开关：TF32 + cuDNN benchmark
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    # DDP 初始化
    local_rank = set_ddp()
    args.local_rank = local_rank
    args.device = f"cuda:{local_rank}"
    args.world_size = dist.get_world_size()
    args.rank = dist.get_rank()
    args.is_main = (args.rank == 0)
    logger(f"ddp 初始化完成，world_size:{args.world_size}，backend=nccl")

    # 打印训练配置
    if args.is_main:
        logger("=" * 60)
        logger("QLoRA参数配置:")
        for k, v in sorted(vars(args).items()):
            logger(f"  {k:30s} = {v}")
        logger("=" * 60)

    # SwanLab 实验跟踪
    if args.use_swanlab and args.is_main:
        swanlab.init(
            project="MyLLM",
            experiment_name=f"DDP-SFT",
            config=args
        )
    
    # 加载 tokenizer 和模型
    random.seed(42)
    np.random.seed(42)    
    torch.manual_seed(42)  # 设置随机种子以确保每个进程加载相同的模型权重
    torch.cuda.manual_seed_all(42)    
    
    tokenizer = AutoTokenizer.from_pretrained(args.load_dir)
    
    # lm_config 不传 → 走 HF 加载分支（args.load_dir）；
    # 若 args.resume 有效 → load_model 内部会优先走 ckpt 分支，并把 ckpt 字典返回出来
    my_model, preloaded_ckpt = load_model(args, dtype=None)
        
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
    ctx = amp.autocast(device_type="cuda", dtype=amp_type)
    scaler = amp.GradScaler(enabled=(args.dtype == "float16"))
    
    # 加载训练数据
    train_ds = SFTDataset(args.data_path)
    lengths = train_ds.get_len()
    train_sampler = DistributedLengthGroupedSampler(dataset=train_ds, 
                                                    batch_size=args.batch_size, 
                                                    lengths=lengths,
                                                    seed=42)
    collator = SFTCollator(tokenizer.pad_token_id)
    data_loader = data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collator
    )
    
    # 优化器 + 断点续训
    optimizer = build_optimizer(my_model, args)
    start_epoch, start_step = maybe_resume(my_model, 
                                           optimizer, 
                                           scaler, 
                                           args, 
                                           ckpt=preloaded_ckpt)
    
    # 训练循环
    for epoch in range(start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)  # 每轮重新洗牌
        epoch_train(
            epoch, my_model, data_loader, optimizer, scaler, ctx, 
            my_model.module.config, args,
            start_step=start_step if epoch == start_epoch else 0,
            ckpt_prefix="sft",
        )

    try:
        if args.is_main:
            save_hfckpt(os.path.join(args.save_dir, f"hf_model_{count_parameters(my_model)}"),
                        my_model, tokenizer)
    finally:
        dist.barrier()

    #销毁ddp环境
    destroy_ddp()