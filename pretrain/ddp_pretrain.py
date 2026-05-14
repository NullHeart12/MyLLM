import os
import argparse
import math
import time
from contextlib import nullcontext

from model import MyModelConfig, Transformer

from transformers import AutoTokenizer
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import amp
from torch.utils import data
from torch import optim
import swanlab

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

def load_tokenizer(tokenizer_path:str):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    return tokenizer

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def load_model(
    lm_config:MyModelConfig,
    args:argparse.Namespace
) -> DDP:
    my_model = Transformer(lm_config)
    my_model = my_model.to(args.device)
    my_model = DDP(
        my_model,
        device_ids=[args.local_rank],
        output_device=args.local_rank,
    )
    
    if args.is_main:
        logger(f"模型参数数量: {count_parameters(my_model) / 1e6:.3f}M")
    
    return my_model

def update_lr(iteration:int, all_iterations:int, warmup_iters:int, base_lr:float):
    min_lr = base_lr * 0.1
    
    if iteration < warmup_iters:
        lr = base_lr * ((iteration + 1 )/ max(1, warmup_iters))
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
    ):
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

        new_lr = update_lr(
            iteration=step + epoch * all_steps,
            all_iterations=all_steps * args.epochs,
            warmup_iters=args.warmup_iters,
            base_lr=args.learning_rate
        )
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr

        # 梯度累积中间步用 no_sync 跳过 DDP all-reduce
        is_accum_step = (step + 1) % args.gradient_accumulation_steps != 0
        sync_ctx = model.no_sync() if is_accum_step else nullcontext()

        with sync_ctx:
            with ctx:
                outputs = model(input_ids=input_ids, labels=labels)
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
                    f"pretrain_param_count{count_parameters(model) / 1e6:.3f}M.pt"
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