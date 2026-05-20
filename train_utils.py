import os
import argparse
import math
import time
from contextlib import nullcontext

import torch
import torch.distributed as dist
from torch import amp, optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils import data
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizerBase
)
import swanlab

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
    args: argparse.Namespace,
    lm_config: MyModelConfig = None,
    dtype: torch.dtype = None,
) -> tuple[DDP, dict | None]:
    """统一的模型加载入口。按优先级判断来源：

      1. args.resume 指向有效 .pt → 从 ckpt 重建模型（config 也从 ckpt 里取）；
      2. lm_config is None         → 从 args.load_dir 的 HuggingFace 目录加载（SFT/DPO 起步）；
      3. lm_config 非 None         → 用 lm_config 随机初始化（pretrain 起步）。

    其余步骤（dtype 转换、to(device)、梯度检查点、DDP 包装、日志）三种来源共用。

    返回 (model, ckpt)：
      - 仅来源 1 时 ckpt 为已加载到 CPU 的字典，**复用给 maybe_resume 避免重复读 .pt**；
      - 来源 2/3 时 ckpt 为 None。
    优化器 / scaler / epoch / step 的恢复仍由 maybe_resume 负责。
    """
    ckpt = None

    # ---- 来源 1：续训 .pt ----
    if getattr(args, "resume", None) and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location="cpu")
        cfg = MyModelConfig(**ckpt["config"])
        my_model = Transformer(cfg)
        my_model.load_state_dict(ckpt["model"])
        if dtype is not None:
            my_model = my_model.to(dtype)
        source_desc = f"从 ckpt {args.resume} 重建"
    # ---- 来源 2：HuggingFace 目录 ----
    elif lm_config is None:
        if getattr(args, "use_auto", False):
            AutoConfig.register("my_model", MyModelConfig)
            AutoModelForCausalLM.register(MyModelConfig, Transformer)
            cls = AutoModelForCausalLM
        else:
            cls = Transformer
        my_model = cls.from_pretrained(args.load_dir, torch_dtype=dtype)
        source_desc = f"从 HF {args.load_dir} 加载"
    # ---- 来源 3：随机初始化 ----
    else:
        my_model = Transformer(lm_config)
        if dtype is not None:
            my_model = my_model.to(dtype)
        source_desc = "随机初始化"

    # ---- 以下三种来源共用 ----
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
        logger(f"{source_desc}，参数量: {count_parameters(my_model) / 1e6:.3f}M")

    return my_model, ckpt


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
                 args: argparse.Namespace,
                 ckpt: dict | None = None) -> tuple[int, int]:
    """恢复训练状态，返回 (start_epoch, start_step)；若没什么可恢复返回 (0, 0)。

    两种调用模式：
      A. 传入预加载的 ckpt（推荐与 `load_model` 配合使用）：
         跳过 .pt 二次读取 + 跳过 model.load_state_dict（假定 load_model 里已经做过），
         只补恢复 optimizer / scaler / epoch / step。
      B. ckpt = None：按 args.resume 自己读 .pt 并完整恢复（model + optimizer + scaler）。
         保留这条路径是为了向后兼容老用法，例如单独脚本只想恢复 optimizer 状态等场景。
    """
    if ckpt is None:
        if args.resume is None or not os.path.isfile(args.resume):
            return 0, 0
        # 老路径：自己读 .pt 并恢复 model 权重
        ckpt = torch.load(args.resume, map_location={"cuda:0": args.device})
        model.module.load_state_dict(ckpt["model"])

    # 不论 ckpt 来自参数还是 args.resume，下面这几步都要做
    optimizer.load_state_dict(ckpt["optimizer"])
    scaler.load_state_dict(ckpt["scaler"])
    start_epoch = ckpt["epoch"]
    start_step  = ckpt["step"]
    if args.is_main:
        logger(f"已恢复训练状态，epoch={start_epoch} step={start_step}")
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


def epoch_train(
                epoch: int,
                model: DDP,
                data_loader: data.DataLoader,
                optimizer: optim.AdamW,
                scaler: amp.GradScaler,
                ctx: amp.autocast,
                lm_config: MyModelConfig,
                args: argparse.Namespace,
                start_step: int = 0,
                ckpt_prefix: str = "ckpt",
    ):
    """单 epoch 训练循环，pretrain / SFT / DPO 共用。

    与具体阶段相关的两点已参数化：
      - ckpt_prefix：保存的 .pt 文件名前缀（pretrain 用 "pretrain"，SFT 用 "sft"，等等）；
      - batch 里的 attention_mask 可选（SFT 变长 collator 会带，pretrain 等长不需要）。
    """
    all_steps = len(data_loader)
    start_time = None  # 在跳过 resume 之前的步骤后再启动计时

    for step, batch in enumerate(data_loader):
        # resume 时跳过已经训过的 step
        if step < start_step:
            continue
        if start_time is None:
            start_time = time.time()

        input_ids = batch['input_ids'].to(args.device)
        labels = batch['labels'].to(args.device)
        # SFT 等变长 batch 会带 attention_mask；pretrain 等长就没有
        attention_mask = batch.get('attention_mask')
        if attention_mask is not None:
            attention_mask = attention_mask.to(args.device)

        new_lr = update_lr(
            iteration=step + epoch * all_steps,
            all_iterations=all_steps * args.epochs,
            warmup_iters=args.warmup_iters,
            # warmup_iters=all_steps * 0.01,
            base_lr=args.learning_rate
        )
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr

        # 梯度累积中间步用 no_sync 跳过 DDP all-reduce
        is_accum_step = (step + 1) % args.gradient_accumulation_steps != 0
        sync_ctx = model.no_sync() if is_accum_step else nullcontext()

        with sync_ctx:
            with ctx:
                outputs = model(
                    input_ids=input_ids,
                    labels=labels,
                    attention_mask=attention_mask,
                )
                loss = outputs.loss / args.gradient_accumulation_steps
            scaler.scale(loss).backward()

        if not is_accum_step:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        if (step + 1) % args.log_interval == 0:
            loss_for_log = loss.detach().clone()
            dist.all_reduce(loss_for_log, op=dist.ReduceOp.AVG)
            if args.is_main:
                display_loss = loss_for_log.item() * args.gradient_accumulation_steps
                spent_time = time.time() - start_time
                # 用"本次运行已跑的步数"做平均，避免 resume 后被绝对步号稀释
                steps_done_this_run = (step + 1) - start_step
                sec_per_step = spent_time / max(1, steps_done_this_run)
                remaining_steps = all_steps - (step + 1)
                elapsed_min = spent_time / 60
                remain_min = remaining_steps * sec_per_step / 60
                total_min = elapsed_min + remain_min
                logger(
                    'Epoch:[{}/{}]({}/{}) loss:{:.3f} lr:{:.7f} elapsed:{:.0f}min remain:{:.0f}min total:{:.0f}min;'.format(
                    epoch + 1,
                    args.epochs,
                    step + 1,
                    all_steps,
                    display_loss,
                    optimizer.param_groups[-1]['lr'],
                    elapsed_min, remain_min, total_min)
                )

                if args.use_swanlab:
                    swanlab.log({
                        "train/loss": display_loss,
                        "train/lr": optimizer.param_groups[-1]['lr']
                    }, step=step + epoch * all_steps)

        if (step + 1) % args.save_interval == 0 and args.is_main:
            model.eval()
            save_path = os.path.join(
                    args.save_dir,
                    f"{ckpt_prefix}_param_count{count_parameters(model) / 1e6:.3f}M.pt"
                )
            save_checkpoint(
                save_path, model, optimizer, scaler, lm_config,
                epoch=epoch, step=step + 1,
                global_step=step + 1 + epoch * all_steps,
            )
            logger(f"模型已保存到 {save_path}")
            model.train()

        if (step + 1) % args.snapshot_interval == 0 and args.is_main:
            model.eval()
            snap_dir = os.path.join(args.save_dir, f"snapshot_{epoch}")
            os.makedirs(snap_dir, exist_ok=True)
            save_path = os.path.join(
                    snap_dir,
                    f"snapshot_step{step + 1}_param_count{count_parameters(model) / 1e6:.3f}M.pt"
                )
            save_checkpoint(
                save_path, model, optimizer, scaler, lm_config,
                epoch=epoch, step=step + 1,
                global_step=step + 1 + epoch * all_steps,
            )
            logger(f"快照已保存到 {save_path}")
            model.train()
