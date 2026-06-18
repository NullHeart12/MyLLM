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
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    StateDictType,
    FullStateDictConfig,
    FullOptimStateDictConfig,
)
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizerBase
)
import swanlab
import mlflow

from model import MyModelConfig, Transformer
from dpo.dpo_loss import compute_dpo_loss, logits_to_log_probs


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


def build_optimizer(model: torch.nn.Module, args: argparse.Namespace) -> optim.AdamW:
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
    del ckpt
    if torch.cuda.is_available():
        torch.cuda.empty_cache()




def trans_to_hfckpt(path_from: str, path_to: str, tokenizer_path: str = None):
    """
    把 save_checkpoint 写出的 .pt 离线转成 HuggingFace 标准目录,便于 from_pretrained 加载。

    参数:
    path_from: 训练 checkpoint 的 .pt 路径(必须由 save_checkpoint 保存,含 "model" 和 "config")
    path_to: 输出目录,会被创建
    tokenizer_path: 可选,若提供则把对应 tokenizer 也一并保存到 path_to
    """
    assert os.path.isfile(path_from), f"checkpoint 不存在: {path_from}"

    ckpt = torch.load(path_from, map_location="cpu", weights_only=False)
    assert "model" in ckpt and "config" in ckpt, \
        "checkpoint 缺少 'model' 或 'config' 字段,无法转换"

    # 提取只需要的两部分到独立变量,然后释放 ckpt（含 optimizer/scaler 等大块内存)
    # 否则后续构造模型时可能触发 OOM（RAM 紧张的机器上特别明显）。
    model_state = ckpt["model"]
    cfg_dict = ckpt["config"]
    del ckpt
    import gc; gc.collect()

    lm_config = MyModelConfig(**cfg_dict)
    model = Transformer(lm_config)

    missing, unexpected = model.load_state_dict(model_state, strict=False)
    del model_state
    gc.collect()
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
    model.module.save_pretrained(path, safe_serialization=True)
    tokenizer.save_pretrained(path)
    logger(f"HF 格式模型已保存到 {path}")


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
    save_fn=None,
):
    """单 epoch 训练循环，pretrain / SFT / DPO / LoRA 共用。

    与具体阶段相关的几点已参数化：
      - ckpt_prefix：保存的 .pt 文件名前缀（pretrain 用 "pretrain"，SFT 用 "sft"，LoRA 用 "lora" 等）；
      - batch 里的 attention_mask 可选（SFT 变长 collator 会带，pretrain 等长不需要）；
      - save_fn：保存函数。None 时走默认 save_checkpoint（保存完整模型权重）；
        传入自定义函数（如 save_lora_checkpoint）则只保存对应内容。签名必须为
        save_fn(path, model, optimizer, scaler, lm_config, epoch, step, global_step)。
    """
    _save_fn = save_fn if save_fn is not None else save_checkpoint
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
                    use_cache=False,
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

                if getattr(args, "use_mlflow", False):
                    mlflow.log_metrics({
                        "train/loss": display_loss,
                        "train/lr":   optimizer.param_groups[-1]['lr'],
                    }, step=step + epoch * all_steps)

        if (step + 1) % args.save_interval == 0 and args.is_main:
            model.eval()
            # LoRA(save_fn=save_lora_checkpoint)用 epoch/step 命名,语义清晰;
            # 全参(默认 save_checkpoint)沿用参数量命名,保持向后兼容
            if save_fn is not None:
                save_path = os.path.join(
                    args.save_dir,
                    f"{ckpt_prefix}_epoch{epoch+1}_step{step+1}.pt"
                )
            else:
                save_path = os.path.join(
                    args.save_dir,
                    f"{ckpt_prefix}_param_count{count_parameters(model) / 1e6:.3f}M.pt"
                )
            _save_fn(
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
            if save_fn is not None:
                save_path = os.path.join(
                    snap_dir,
                    f"snapshot_{ckpt_prefix}_step{step+1}.pt"
                )
            else:
                save_path = os.path.join(
                    snap_dir,
                    f"snapshot_step{step + 1}_param_count{count_parameters(model) / 1e6:.3f}M.pt"
                )
            _save_fn(
                save_path, model, optimizer, scaler, lm_config,
                epoch=epoch, step=step + 1,
                global_step=step + 1 + epoch * all_steps,
            )
            logger(f"快照已保存到 {save_path}")
            model.train()

def _model_log_probs(model, input_ids, labels, attention_mask):
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
    )
    return logits_to_log_probs(outputs.logits, labels)


def save_dpo_fsdp_checkpoint(
    path: str,
    policy_model: torch.nn.Module,
    optimizer: optim.Optimizer,
    scaler: amp.GradScaler,
    epoch: int,
    step: int,
    global_step: int,
):
    """保存 FSDP DPO 训练 checkpoint，不导出 HuggingFace 格式。"""
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    is_main = rank == 0

    full_state_config = FullStateDictConfig(
        offload_to_cpu=True,
        rank0_only=True,
    )
    full_optim_config = FullOptimStateDictConfig(
        offload_to_cpu=True,
        rank0_only=True,
    )

    with FSDP.state_dict_type(
        policy_model,
        StateDictType.FULL_STATE_DICT,
        full_state_config,
        full_optim_config,
    ):
        model_state = policy_model.state_dict()
        optimizer_state = FSDP.optim_state_dict(policy_model, optimizer)

    if is_main:
        os.makedirs(path, exist_ok=True)
        torch.save(model_state, os.path.join(path, "model_state.pt"))
        torch.save(
            {
                "optimizer": optimizer_state,
                "scaler": scaler.state_dict(),
                "epoch": epoch,
                "step": step,
                "global_step": global_step,
            },
            os.path.join(path, "trainer_state.pt"),
        )

    del model_state, optimizer_state
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def save_dpo_fsdp_hf(
    path: str,
    policy_model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
):
    """将 FSDP DPO policy model 导出为 HuggingFace 格式，并保存 tokenizer。"""
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    is_main = rank == 0
    full_state_config = FullStateDictConfig(
        offload_to_cpu=True,
        rank0_only=True,
    )

    with FSDP.state_dict_type(
        policy_model,
        StateDictType.FULL_STATE_DICT,
        full_state_config,
    ):
        model_state = policy_model.state_dict()

    if is_main:
        os.makedirs(path, exist_ok=True)
        policy_model.module.save_pretrained(
            path,
            state_dict=model_state,
            safe_serialization=True,
        )
        tokenizer.save_pretrained(path)
        logger(f"HF 格式 DPO 模型已保存到 {path}")

    del model_state
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def maybe_resume_dpo_fsdp(
    policy_model: torch.nn.Module,
    optimizer: optim.Optimizer,
    scaler: amp.GradScaler,
    args: argparse.Namespace,
) -> tuple[int, int]:
    """恢复 FSDP DPO 训练状态，返回 (start_epoch, start_step)。"""
    resume_dir = getattr(args, "resume", None)
    if resume_dir is None:
        return 0, 0

    model_path = os.path.join(resume_dir, "model_state.pt")
    trainer_state_path = os.path.join(resume_dir, "trainer_state.pt")
    if not os.path.isfile(model_path) or not os.path.isfile(trainer_state_path):
        if getattr(args, "is_main", False):
            logger(f"[warn] resume 路径缺少 model_state.pt 或 trainer_state.pt: {resume_dir}")
        dist.barrier()
        return 0, 0

    full_state_config = FullStateDictConfig(
        offload_to_cpu=True,
        rank0_only=False,
    )
    full_optim_config = FullOptimStateDictConfig(
        offload_to_cpu=True,
        rank0_only=False,
    )

    model_state = torch.load(model_path, map_location="cpu", weights_only=True)
    trainer_state = torch.load(trainer_state_path, map_location="cpu", weights_only=False)

    with FSDP.state_dict_type(
        policy_model,
        StateDictType.FULL_STATE_DICT,
        full_state_config,
        full_optim_config,
    ):
        policy_model.load_state_dict(model_state)
        optim_state = FSDP.optim_state_dict_to_load(
            policy_model,
            optimizer,
            trainer_state["optimizer"],
        )

    optimizer.load_state_dict(optim_state)
    scaler.load_state_dict(trainer_state["scaler"])

    start_epoch = trainer_state["epoch"]
    start_step = trainer_state["step"]
    if getattr(args, "is_main", False):
        logger(f"已恢复 DPO FSDP 训练状态: epoch={start_epoch}, step={start_step}")

    del model_state, trainer_state, optim_state
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    dist.barrier()
    return start_epoch, start_step

def epoch_train_dpo_fsdp(
    epoch: int,
    policy_model: torch.nn.Module,
    ref_model: torch.nn.Module,
    data_loader: data.DataLoader,
    optimizer: optim.AdamW,
    scaler: amp.GradScaler,
    ctx: amp.autocast,
    args: argparse.Namespace,
    start_step: int = 0,
    ckpt_prefix: str = "dpo",
    save_fn=None,
):
    """单 epoch DPO 训练循环，面向 FSDP policy/ref model。

    DPO batch 字段：
      chosen_ids / chosen_labels / chosen_attention_mask
      rejected_ids / rejected_labels / rejected_attention_mask

    保存函数默认不执行；FSDP checkpoint 需要外部传入专用 save_fn。
    """
    _save_fn = save_fn if save_fn is not None else save_dpo_fsdp_checkpoint
    all_steps = len(data_loader)
    start_time = None
    log_sums = torch.zeros(4, device=args.device, dtype=torch.float32)

    for step, batch in enumerate(data_loader):
        if step < start_step:
            continue
        if start_time is None:
            start_time = time.time()

        chosen_ids = batch['chosen_ids'].to(args.device)
        chosen_labels = batch['chosen_labels'].to(args.device)
        chosen_attention_mask = batch['chosen_attention_mask'].to(args.device)
        rejected_ids = batch['rejected_ids'].to(args.device)
        rejected_labels = batch['rejected_labels'].to(args.device)
        rejected_attention_mask = batch['rejected_attention_mask'].to(args.device)

        new_lr = update_lr(
            iteration=step + epoch * all_steps,
            all_iterations=all_steps * args.epochs,
            warmup_iters=args.warmup_iters,
            base_lr=args.learning_rate,
        )
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr

        is_accum_step = (step + 1) % args.gradient_accumulation_steps != 0
        sync_ctx = policy_model.no_sync() if is_accum_step else nullcontext()

        with sync_ctx:
            with ctx:
                policy_chosen_log_probs = _model_log_probs(
                    policy_model, chosen_ids, chosen_labels, chosen_attention_mask
                )
                policy_rejected_log_probs = _model_log_probs(
                    policy_model, rejected_ids, rejected_labels, rejected_attention_mask
                )

                with torch.no_grad():
                    ref_chosen_log_probs = _model_log_probs(
                        ref_model, chosen_ids, chosen_labels, chosen_attention_mask
                    )
                    ref_rejected_log_probs = _model_log_probs(
                        ref_model, rejected_ids, rejected_labels, rejected_attention_mask
                    )

                dpo_loss = compute_dpo_loss(
                    ref_chosen_log_probs=ref_chosen_log_probs,
                    ref_rejected_log_probs=ref_rejected_log_probs,
                    policy_chosen_log_probs=policy_chosen_log_probs,
                    policy_rejected_log_probs=policy_rejected_log_probs,
                    beta=args.beta,
                )
                loss = dpo_loss / args.gradient_accumulation_steps

            scaler.scale(loss).backward()

        if not is_accum_step:
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            policy_model.clip_grad_norm_(args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            policy_chosen_seq = policy_chosen_log_probs.sum(dim=1)
            policy_rejected_seq = policy_rejected_log_probs.sum(dim=1)
            ref_chosen_seq = ref_chosen_log_probs.sum(dim=1)
            ref_rejected_seq = ref_rejected_log_probs.sum(dim=1)
            chosen_rewards = args.beta * (policy_chosen_seq - ref_chosen_seq)
            rejected_rewards = args.beta * (policy_rejected_seq - ref_rejected_seq)
            reward_margin = chosen_rewards - rejected_rewards
            n_samples = reward_margin.numel()
            log_sums += torch.stack([
                dpo_loss.detach().float() * n_samples,
                reward_margin.detach().float().sum(),
                (reward_margin > 0).float().sum(),
                torch.tensor(float(n_samples), device=args.device),
            ])

        if (step + 1) % args.log_interval == 0:
            metrics = log_sums.clone()
            dist.all_reduce(metrics, op=dist.ReduceOp.SUM)

            if args.is_main:
                n_logged = max(1.0, metrics[3].item())
                display_loss = (metrics[0] / n_logged).item()
                display_margin = (metrics[1] / n_logged).item()
                display_acc = (metrics[2] / n_logged).item()
                spent_time = time.time() - start_time
                steps_done_this_run = (step + 1) - start_step
                sec_per_step = spent_time / max(1, steps_done_this_run)
                remaining_steps = all_steps - (step + 1)
                elapsed_min = spent_time / 60
                remain_min = remaining_steps * sec_per_step / 60
                total_min = elapsed_min + remain_min
                logger(
                    'Epoch:[{}/{}]({}/{}) dpo_loss:{:.4f} margin:{:.4f} acc:{:.3f} '
                    'lr:{:.7f} elapsed:{:.0f}min remain:{:.0f}min total:{:.0f}min;'.format(
                        epoch + 1,
                        args.epochs,
                        step + 1,
                        all_steps,
                        display_loss,
                        display_margin,
                        display_acc,
                        optimizer.param_groups[-1]['lr'],
                        elapsed_min, remain_min, total_min,
                    )
                )

                if getattr(args, "use_swanlab", False):
                    swanlab.log({
                        "train/dpo_loss": display_loss,
                        "train/reward_margin": display_margin,
                        "train/preference_acc": display_acc,
                        "train/lr": optimizer.param_groups[-1]['lr'],
                    }, step=step + epoch * all_steps)

                if getattr(args, "use_mlflow", False):
                    mlflow.log_metrics({
                        "train/dpo_loss": display_loss,
                        "train/reward_margin": display_margin,
                        "train/preference_acc": display_acc,
                        "train/lr": optimizer.param_groups[-1]['lr'],
                    }, step=step + epoch * all_steps)

            log_sums.zero_()

        if (step + 1) % args.save_interval == 0:
            policy_model.eval()
            save_path = os.path.join(
                args.save_dir,
                f"{ckpt_prefix}_epoch{epoch+1}_step{step+1}",
            )
            _save_fn(
                save_path, policy_model, optimizer, scaler, epoch,
                step + 1, step + 1 + epoch * all_steps,
            )
            if args.is_main:
                logger(f"DPO checkpoint 已保存到 {save_path}")
            dist.barrier()
            policy_model.train()
            ref_model.eval()

        snapshot_interval = getattr(args, "snapshot_interval", 0)
        if snapshot_interval > 0 and (step + 1) % snapshot_interval == 0:
            policy_model.eval()
            snap_dir = os.path.join(args.save_dir, f"snapshot_{epoch}")
            save_path = os.path.join(
                snap_dir,
                f"snapshot_{ckpt_prefix}_step{step+1}",
            )
            _save_fn(
                save_path, policy_model, optimizer, scaler, epoch,
                step + 1, step + 1 + epoch * all_steps,
            )
            if args.is_main:
                logger(f"DPO snapshot 已保存到 {save_path}")
            dist.barrier()
            policy_model.train()
            ref_model.eval()
