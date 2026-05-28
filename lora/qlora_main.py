import os
import argparse
import random
import numpy as np
from contextlib import nullcontext

from train_utils import (
    set_ddp, destroy_ddp, logger,
    epoch_train, count_parameters
)
from deal_dataset.dataset import PretrainDataset
from .model_lora import (
    apply_lora, save_lora, load_lora,
    save_lora_checkpoint, load_lora_checkpoint,
)

import swanlab
import torch
import torch.distributed as dist
from torch import amp
from torch.utils import data
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,             # 把一个 module 包成"自动 GC 的 module"
    CheckpointImpl,                 # NO_REENTRANT / REENTRANT 二选一
    apply_activation_checkpointing, # 批量替换:扫遍模型,符合条件的 module 整个换成 wrapped 版
)
from functools import partial
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import prepare_model_for_kbit_training
import bitsandbytes as bnb


if __name__=="__main__":
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # PROJECT_ROOT = "/root/autodl-tmp/MyLLMDataset"
    
    parser = argparse.ArgumentParser(description="Train QWen3.6-27B with QLoRA")
    
    # 基础训练参数
    parser.add_argument("--name_or_path", type=str, 
                        default="Qwen/Qwen3.6-27B",
                        help="Qwen3.6-27B 模型路径或名称（HF Hub）")
    parser.add_argument("--out_dir", type=str, 
                        default=os.path.join(PROJECT_ROOT, "model", "QLoRA_model"), 
                        help="模型输出目录")
    
    parser.add_argument("--epochs", type=int, default=2, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=1, 
                        help="每张卡的批次大小（DDP 约定）。全局有效 batch = batch_size × world_size × gradient_accumulation_steps")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="学习率")

    # 实验跟踪和数据加载参数
    parser.add_argument("--use_swanlab", action="store_true", help="是否使用SwanLab进行实验跟踪")
    parser.add_argument("--num_workers", type=int, default=4, help="数据加载的工作进程数")
    parser.add_argument("--data_path", type=str, 
                        # default=os.path.join(PROJECT_ROOT, "processed_dataset", 
                                            #  "BelleGroup_sft.jsonl"), 
                        default=os.path.join(PROJECT_ROOT, "processed_dataset", 
                                             "poetry_arrow"), 
                        help="训练数据路径")

    # 训练优化参数
    parser.add_argument("--gradient_accumulation_steps",
                        type=int, default=16, help="梯度累积步数")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        help="启用梯度检查点：以重算前向为代价，按 TransformerBlock 粒度减少显存占用")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--warmup_iters", type=int, default=100, help="学习率预热迭代次数")
    parser.add_argument("--weight_decay", type=float, default=0.1, 
                        help="AdamW weight decay")
    parser.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.95), 
                        help="AdamW betas")
    parser.add_argument("--lora_rank", type=int, default=8, help="LoRA 低秩矩阵的秩")
    parser.add_argument("--lora_alpha", type=int, default=8, help="LoRA 的 alpha 超参数")
    parser.add_argument("--lora_dropout", type=float, default=0.0, help="LoRA dropout 概率")

    # 日志和保存参数
    parser.add_argument("--log_interval", type=int, default=150, help="日志记录间隔")
    # epoch_train 内部按 save_interval 调 save_lora_checkpoint(只存 adapter+optimizer 等,不存量化 base)
    parser.add_argument("--save_interval", type=int, default=2000, help="LoRA 断点保存间隔(step)")
    parser.add_argument("--snapshot_interval", type=int, default=10**12, help="LoRA 快照保存间隔(step),默认禁用")

    # 断点续训(LoRA checkpoint:权重 + optimizer + 进度)
    parser.add_argument("--resume", type=str, default=None,
                        help="从指定 LoRA checkpoint (.pt) 完整恢复(权重+optimizer+进度)")

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
            experiment_name=f"DDP-QLoRA",
            config=args
        )
    
    # 加载 tokenizer 和模型
    random.seed(42)
    np.random.seed(42)    
    torch.manual_seed(42)  # 设置随机种子以确保每个进程加载相同的模型权重
    torch.cuda.manual_seed_all(42)    
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.name_or_path
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.name_or_path,
        quantization_config=bnb_config,
        device_map={"": args.local_rank},
    )
    model = prepare_model_for_kbit_training(
        model, 
        use_gradient_checkpointing=False,
    )
    apply_lora(
        model,
        target_suffixes=(
            "q_proj", "k_proj", "v_proj", "o_proj",     # 注意力层
            "gate_proj", "up_proj", "down_proj",        # MLP 层
        ),
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
    )
    
    # 训练参数量打印(只数可训练的 LoRA 参数)
    if args.is_main:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in model.parameters())
        logger(f"可训练参数: {trainable/1e6:.3f}M / 总参数: {total/1e6:.3f}M "
               f"(占比 {100*trainable/total:.4f}%)")

    if args.gradient_checkpointing:
        DecoderLayerCls = type(model.model.layers[0])
        
        apply_activation_checkpointing(
            model,
            checkpoint_wrapper_fn=partial(
                checkpoint_wrapper,
                checkpoint_impl=CheckpointImpl.NO_REENTRANT,   
            ),
            check_fn=lambda m: isinstance(m, DecoderLayerCls),
        )
        if args.is_main:
            logger(f"已对 {DecoderLayerCls.__name__} 手动套 activation checkpointing")

    # DDP 包装 —— 量化 base 参数 requires_grad=False,DDP 默认只同步 trainable(LoRA)。
    # find_unused_parameters=False:LoRA 层每步都走到,不会有 unused 参数。
    model = DDP(
        model,
        device_ids=[args.local_rank],
        output_device=args.local_rank,
        find_unused_parameters=False,
    )

    # 训练准备
    if args.is_main:
        args.save_dir = os.path.join(args.out_dir)
        os.makedirs(args.save_dir, exist_ok=True)
    dist.barrier()  # 确保所有进程都完成了模型加载和准备

    # QLoRA 不需要外层 autocast/GradScaler，这里保留是为了复用epoch_train()函数:
    # - compute dtype 已经在 BitsAndBytesConfig.bnb_4bit_compute_dtype 里指定为 bf16
    # - LoRA dtype 跟随 compute dtype(LoRALinear 自动检测)
    # - bf16 不需要 GradScaler(只有 fp16 才需要)
    ctx = nullcontext()
    scaler = amp.GradScaler(enabled=False)

    # 加载训练数据(continued pretraining 风格:样本已被 PretrainProcessor 预先 chunk 成等长)
    # 因此不需要 collator、不需要按长度分桶,默认 DistributedSampler + default_collate 即可
    train_ds = PretrainDataset(args.data_path)
    train_sampler = data.distributed.DistributedSampler(train_ds, shuffle=True)
    data_loader = data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # 优化器:只优化 LoRA 参数,用 8bit Adam 省显存
    # LoRA 参数都是 2D 矩阵,统一上 weight decay 即可,不用像全参 SFT 那样分组
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = bnb.optim.PagedAdamW8bit(
        trainable_params,
        lr=args.learning_rate,
        betas=tuple(args.betas),
        weight_decay=args.weight_decay,
        eps=1e-8,
    )

    # 断点续训(必须在 optimizer 构建后)
    start_epoch, start_step = load_lora_checkpoint(model, optimizer, scaler, args)
    if args.is_main and args.resume is not None and os.path.isfile(args.resume):
        logger(f"已从 {args.resume} 恢复:epoch={start_epoch}, step={start_step}")

    # 训练循环:save_fn=save_lora_checkpoint 让 epoch_train 内部 save_interval 触发时
    # 只存 LoRA adapter + optimizer 进度,不会 dump 4bit 量化 base
    for epoch in range(start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)
        epoch_train(
            epoch, model, data_loader, optimizer, scaler, ctx,
            model.module.config, args,
            start_step=start_step if epoch == start_epoch else 0,
            ckpt_prefix="qlora",
            save_fn=save_lora_checkpoint,
        )

        # 每轮结束多存一份纯 adapter(无 optimizer state,方便部署/分享)
        if args.is_main:
            adapter_path = os.path.join(
                args.save_dir, f"qlora_adapter_epoch{epoch+1}.pt"
            )
            save_lora(model, adapter_path)
            logger(f"LoRA adapter 已保存到 {adapter_path}")

    # 最终 adapter 保存
    try:
        if args.is_main:
            final_path = os.path.join(args.save_dir, "qlora_adapter_final.pt")
            save_lora(model, final_path)
            logger(f"最终 LoRA adapter 已保存到 {final_path}")
            tokenizer.save_pretrained(args.save_dir)
    finally:
        dist.barrier()

    destroy_ddp()