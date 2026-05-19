import os
import argparse
import math

import torch
import torch.distributed as dist
from torch import amp, optim
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from model import MyModelConfig, Transformer


def set_ddp():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def destroy_ddp():
    dist.destroy_process_group()


def logger(content: str, force: bool = False):
    """只在 rank 0 打印；DDP 未初始化时按 LOCAL_RANK 兜底。"""
    if dist.is_available() and dist.is_initialized():
        if dist.get_rank() == 0 or force:
            print(content)
        return
    if int(os.environ.get("LOCAL_RANK", "0")) == 0 or force:
        print(content)


def load_tokenizer(tokenizer_path: str) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    return tokenizer


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_model(
    lm_config: MyModelConfig,
    args: argparse.Namespace,
) -> DDP:
    my_model = Transformer(lm_config)
    my_model = my_model.to(args.device)

    # 梯度检查点：必须在 DDP 包裹前启用。
    # use_reentrant=False 走 saved_tensors_hooks 路径，反向时 DDP reducer 的 hook 注册/触发
    # 与普通训练一致，所以这里 DDP 不需要 static_graph / find_unused_parameters；
    # 反过来 static_graph=True 会与训练循环里的 model.no_sync()(梯度累积) 冲突。
    use_gc = getattr(args, "gradient_checkpointing", False)
    if use_gc:
        my_model.gradient_checkpointing_enable(use_reentrant=False)
        if args.is_main:
            logger("已启用 gradient checkpointing (use_reentrant=False)")

    my_model = DDP(
        my_model,
        device_ids=[args.local_rank],
        output_device=args.local_rank,
    )

    if args.is_main:
        logger(f"模型参数数量: {count_parameters(my_model) / 1e6:.3f}M")

    return my_model


def update_lr(iteration: int, all_iterations: int, warmup_iters: int, base_lr: float):
    min_lr = base_lr * 0.1

    if iteration < warmup_iters:
        lr = base_lr * ((iteration + 1) / max(1, warmup_iters))
    elif iteration > all_iterations:
        lr = min_lr
    else:
        ratio = (iteration - warmup_iters) / (all_iterations - warmup_iters)
        lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * ratio))

    return lr


def build_optimizer(model: DDP, args: argparse.Namespace) -> optim.AdamW:
    """构造 AdamW，并把 norm/bias/embedding 这类参数排除在 weight decay 之外。"""
    decay_params, no_decay_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2 or n.endswith(".bias") or "RMSNorm" in n or "embeddings" in n:
            no_decay_params.append(p)
        else:
            decay_params.append(p)
    return optim.AdamW(
        [{"params": decay_params,    "weight_decay": args.weight_decay},
         {"params": no_decay_params, "weight_decay": 0.0}],
        lr=args.learning_rate,
        betas=tuple(args.betas),
        eps=1e-8,
    )


def maybe_resume(model: DDP,
                 optimizer: optim.Optimizer,
                 scaler: amp.GradScaler,
                 args: argparse.Namespace) -> tuple[int, int]:
    """
    如果 args.resume 指向有效 checkpoint，则恢复 model/optimizer/scaler 状态，
    返回 (start_epoch, start_step)；否则返回 (0, 0)。
    """
    if args.resume is None or not os.path.isfile(args.resume):
        return 0, 0

    map_location = {"cuda:0": args.device}
    ckpt = torch.load(args.resume, map_location=map_location)
    model.module.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scaler.load_state_dict(ckpt["scaler"])
    start_epoch = ckpt["epoch"]
    start_step  = ckpt["step"]
    if args.is_main:
        logger(f"已从 {args.resume} 恢复，epoch={start_epoch} step={start_step}")
    dist.barrier()
    return start_epoch, start_step


def save_checkpoint(path, model, optimizer, scaler, lm_config,
                    epoch, step, global_step):
    ckpt = {
        "model":       model.module.state_dict(),
        "optimizer":   optimizer.state_dict(),
        "scaler":      scaler.state_dict(),
        "epoch":       epoch,
        "step":        step,
        "global_step": global_step,
        "config":      lm_config.to_dict(),
    }
    torch.save(ckpt, path)


def trans_to_hfckpt(path_from: str, path_to: str, tokenizer_path: str = None):
    """
    把 save_checkpoint 写出的 .pt 离线转成 HuggingFace 标准目录,便于 from_pretrained 加载。

    参数:
    path_from: 训练 checkpoint 的 .pt 路径(必须由 save_checkpoint 保存,含 "model" 和 "config")
    path_to: 输出目录,会被创建
    tokenizer_path: 可选,若提供则把对应 tokenizer 也一并保存到 path_to
    """
    assert os.path.isfile(path_from), f"checkpoint 不存在: {path_from}"

    ckpt = torch.load(path_from, map_location="cpu")
    assert "model" in ckpt and "config" in ckpt, \
        "checkpoint 缺少 'model' 或 'config' 字段,无法转换"

    lm_config = MyModelConfig(**ckpt["config"])
    model = Transformer(lm_config)

    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    if missing:
        logger(f"[warn] 缺失参数: {missing}", force=True)
    if unexpected:
        logger(f"[warn] 多余参数: {unexpected}", force=True)
    model.eval()

    os.makedirs(path_to, exist_ok=True)
    model.save_pretrained(path_to, safe_serialization=True)

    if tokenizer_path is not None:
        tokenizer = load_tokenizer(tokenizer_path)
        tokenizer.save_pretrained(path_to)

    logger(f"已将 {path_from} 转为 HF 格式 -> {path_to}", force=True)


def save_hfckpt(path: str, model: DDP, tokenizer: PreTrainedTokenizerBase):
    hf_dir = os.path.join(path, "hf_model")
    model.module.save_pretrained(hf_dir, safe_serialization=True)
    tokenizer.save_pretrained(hf_dir)
    logger(f"HF 格式模型已保存到 {hf_dir}")
