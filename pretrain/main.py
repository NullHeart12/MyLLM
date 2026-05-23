import os
import argparse
import random
import numpy as np

from model import MyModelConfig
from train_utils import (
    set_ddp, destroy_ddp, logger,
    load_tokenizer, load_model,
    build_optimizer, maybe_resume,
    save_hfckpt, epoch_train,
    count_parameters
)
from deal_dataset.dataset import PretrainDataset

import swanlab
import torch
import torch.distributed as dist
from torch import amp
from torch.utils import data

if __name__ == "__main__":
    # PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    PROJECT_ROOT = "/root/autodl-tmp/MyLLMDataset"
    
    parser = argparse.ArgumentParser(description="using DDP Pretrain MyLLM")
    
    # 基础训练参数
    parser.add_argument("--out_dir", type=str, 
                        default=os.path.join(PROJECT_ROOT, "base_model"), 
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
                        # default=os.path.join(PROJECT_ROOT, "processed_dataset", "seq_monkey.jsonl"), 
                        default=os.path.join(PROJECT_ROOT, "processed_dataset", "seq_monkey_arrow"), 
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
    parser.add_argument("--snapshot_interval", type=int, default=50000, help="生成快照间隔")

    # 模型结构参数（对应 MyModelConfig）
    parser.add_argument("--dim", type=int, default=1024, help="hidden dim")
    parser.add_argument("--n_layers", type=int, default=28, help="Transformer 层数")
    parser.add_argument("--n_heads", type=int, default=16, help="多头注意力头数（dim 必须能被它整除）")
    parser.add_argument("--n_kv_heads", type=int, default=4, help="KV 头数（GQA；n_heads 必须能被它整除）")
    parser.add_argument("--hidden_dim", type=int, default=None,
                        help="FFN 隐层维度；None 时按 2/3 * 4 * dim 自动计算，并对齐 multiple_of")
    parser.add_argument("--multiple_of", type=int, default=64, help="FFN hidden_dim 对齐倍数")
    parser.add_argument("--max_seq_len", type=int, default=1024, help="最大序列长度")
    parser.add_argument("--norm_eps", type=float, default=1e-5, help="RMSNorm epsilon")
    parser.add_argument("--model_dropout", type=float, default=0.0,
                        help="模型内部 dropout 率")
    parser.add_argument("--flash_attention", action="store_true", default=True,
                        help="启用 scaled_dot_product_attention(Flash Attn)")
    parser.add_argument("--no_flash_attention", dest="flash_attention", action="store_false",
                        help="关闭 Flash Attention（fallback 到手写 attention）")

    # MoE 参数
    parser.add_argument("--use_moe", action="store_true", help="启用 MoE FFN")
    parser.add_argument("--n_experts", type=int, default=8, help="MoE 专家数")
    parser.add_argument("--moe_top_k", type=int, default=3, help="MoE 每 token 路由到的专家数")
    parser.add_argument("--router_aux_loss_coef", type=float, default=1e-2,
                        help="MoE 负载均衡辅助损失系数")

    # 断点续训
    parser.add_argument("--resume", type=str,
                        default=None,
                        # default=os.path.join(PROJECT_ROOT, "base_model", "pretrain_param_count82.595M.pt"),
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
        logger("训练参数配置:")
        for k, v in sorted(vars(args).items()):
            logger(f"  {k:30s} = {v}")
        logger("=" * 60)

    # SwanLab 实验跟踪
    if args.use_swanlab and args.is_main:
        swanlab.init(
            project="MyLLM",
            experiment_name=f"DDP-Pretrain",
            config=args
        )
    
    # 加载 tokenizer 和模型
    random.seed(42)
    np.random.seed(42)    
    torch.manual_seed(42)  # 设置随机种子以确保每个进程加载相同的模型权重
    torch.cuda.manual_seed_all(42)

    tokenizer = load_tokenizer(os.path.join(PROJECT_ROOT, "tokenizer_k"))
    
    lm_config = MyModelConfig(
        dim=args.dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        n_kv_heads=args.n_kv_heads,
        vocab_size=tokenizer.vocab_size,
        hidden_dim=args.hidden_dim,
        multiple_of=args.multiple_of,
        max_seq_len=args.max_seq_len,
        norm_eps=args.norm_eps,
        dropout=args.model_dropout,
        flash_attention=args.flash_attention,
        use_moe=args.use_moe,
        n_experts=args.n_experts,
        moe_top_k=args.moe_top_k,
        router_aux_loss_coef=args.router_aux_loss_coef,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    my_model, preloaded_ckpt = load_model(args, lm_config=lm_config)
    
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
    train_ds = PretrainDataset(args.data_path)
    train_sampler = data.distributed.DistributedSampler(train_ds, shuffle=True)
    data_loader = data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
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
            epoch, my_model, data_loader, optimizer, scaler, ctx, lm_config, args,
            start_step=start_step if epoch == start_epoch else 0,
            ckpt_prefix="pretrain",
        )

    try:
        if args.is_main:
            save_hfckpt(os.path.join(args.save_dir, 
                                     f"hf_model_{count_parameters(my_model)}"), 
                        my_model, tokenizer)
    finally:
        dist.barrier()

    #销毁ddp环境
    destroy_ddp()