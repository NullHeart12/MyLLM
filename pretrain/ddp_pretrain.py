import os
import argparse
import time
from contextlib import nullcontext

import torch
import torch.distributed as dist
from torch import amp, optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils import data
import swanlab

from model import MyModelConfig
from train_utils import (
    update_lr,
    logger,
    count_parameters,
    save_checkpoint,
)


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
