import argparse
import os
import sys
import random
import numpy as np

from deal_dataset.dataset import DPODataset, DPOCollator
from train_utils import (
    build_optimizer,
    epoch_train_dpo_fsdp,
    maybe_resume_dpo_fsdp,
    save_dpo_fsdp_hf,
)

import torch
from torch import amp
from torch import distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    MixedPrecision,
    BackwardPrefetch,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from functools import partial
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.trainer_pt_utils import DistributedLengthGroupedSampler
import swanlab

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DPO training with FSDP")

    # Paths
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument(
        "--data_path",
        "--train_data_path",
        dest="data_path",
        type=str,
        required=True,
        help="Tokenized DPO dataset path, usually an Arrow directory.",
    )
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--resume", type=str, default=None, help="DPO FSDP checkpoint directory")

    # Training
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=5e-7)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--max_len", type=int, default=4096)
    parser.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--warmup_iters", type=int, default=100)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.95))

    # FSDP
    parser.add_argument(
        "--fsdp_sharding_strategy",
        type=str,
        default="full_shard",
        choices=["full_shard", "shard_grad_op", "no_shard", "hybrid_shard"],
    )
    parser.add_argument("--gradient_checkpointing", action="store_true")

    # Logging / saving
    parser.add_argument("--use_swanlab", action="store_true", help="是否使用 SwanLab 进行实验跟踪")
    parser.add_argument("--swanlab_run_name", type=str, default=None, help="SwanLab run name")
    parser.add_argument("--save_interval", type=int, default=500)
    parser.add_argument("--snapshot_interval", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=10)

    # DataLoader
    parser.add_argument("--num_workers", type=int, default=4)
    
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    
    random.seed(42)
    np.random.seed(42)    
    torch.manual_seed(42)  # 设置随机种子，确保每个进程加载相同的模型权重
    torch.cuda.manual_seed_all(42)
    
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    
    device = torch.device("cuda", local_rank)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    is_main = rank == 0
    
    args.local_rank = local_rank
    args.world_size = world_size
    args.device = device
    args.rank = rank
    args.is_main = is_main
    
    # 性能开关：TF32 + cuDNN benchmark
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    # AMP autocast + GradScaler：bf16 通常不需要 scaler，fp16 才启用。
    amp_type = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    ctx = amp.autocast(device_type="cuda", dtype=amp_type)
    scaler = amp.GradScaler(enabled=(args.dtype == "float16"))
    
    args.save_dir = os.path.join(args.out_dir)
    if args.is_main:
        os.makedirs(args.save_dir, exist_ok=True)
    dist.barrier()  # 确保所有进程都完成了保存目录创建
    
    if is_main:
        print("=" * 80)
        print("DPO FSDP config:")
        for key, value in sorted(vars(args).items()):
            print(f"{key:32s}: {value}")
        print("=" * 80)
    
    # SwanLab 实验跟踪
    if args.use_swanlab and args.is_main:
        swanlab_run_name = args.swanlab_run_name or (
            f"FSDP-DPO-{args.fsdp_sharding_strategy}"
            f"-beta{args.beta}-lr{args.learning_rate}"
        )
        swanlab.init(
            project="MyLLM",
            experiment_name=swanlab_run_name,
            config=args
        )
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    policy_model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=amp_type,
    )
    ref_model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=amp_type,
    )
    
    policy_model.config.use_cache = False
    ref_model.config.use_cache = False
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad_(False)
    
    if args.gradient_checkpointing:
        policy_model.gradient_checkpointing_enable(use_reentrant=False)
    
    sharding_strategy_map = {
        "full_shard": ShardingStrategy.FULL_SHARD,
        "shard_grad_op": ShardingStrategy.SHARD_GRAD_OP,
        "no_shard": ShardingStrategy.NO_SHARD,
        "hybrid_shard": ShardingStrategy.HYBRID_SHARD,
    }
    mixed_precision = MixedPrecision(
        param_dtype=amp_type,
        reduce_dtype=amp_type,
        buffer_dtype=amp_type,
    )
    
    transformer_layer_cls = {policy_model.model.layers[0].__class__}
    if is_main:
        layer_names = ", ".join(cls.__name__ for cls in transformer_layer_cls)
        print(f"FSDP auto wrap transformer layers: {layer_names}")

    auto_wrap_policy = partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls=transformer_layer_cls,
    )
    
    policy_model = FSDP(
        policy_model,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=mixed_precision,
        sharding_strategy=sharding_strategy_map[args.fsdp_sharding_strategy],
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        device_id=device,
        use_orig_params=True,
    )
    ref_model = FSDP(
        ref_model,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=mixed_precision,
        sharding_strategy=sharding_strategy_map[args.fsdp_sharding_strategy],
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        device_id=device,
        use_orig_params=True,
    )
    
    train_dataset = DPODataset(args.data_path)
    lengths = train_dataset.get_len()
    train_sampler = DistributedLengthGroupedSampler(
        dataset=train_dataset,
        batch_size=args.batch_size,
        lengths=lengths,
        seed=42,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        collate_fn=DPOCollator(tokenizer.pad_token_id),
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    
    optimizer = build_optimizer(policy_model, args)
    start_epoch, start_step = maybe_resume_dpo_fsdp(
        policy_model,
        optimizer,
        scaler,
        args,
    )
    
    for epoch in range(start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)
        policy_model.train()
        ref_model.eval()
        epoch_start_step = start_step if epoch == start_epoch else 0
        
        epoch_train_dpo_fsdp(
            epoch=epoch,
            policy_model=policy_model,
            ref_model=ref_model,
            data_loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            ctx=ctx,
            args=args,
            start_step=epoch_start_step,
        )
    
    policy_model.eval()
    save_dpo_fsdp_hf(
        os.path.join(args.save_dir, "final_hf"),
        policy_model,
        tokenizer,
    )
    
    dist.barrier()
    dist.destroy_process_group()
    
