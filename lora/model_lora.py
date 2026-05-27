import math
import os

import torch
from torch import nn

class LoRALinear(nn.Module):
    def __init__(self,
                 base: nn.Linear,
                 rank: int=8,
                 alpha: int=16,
                 dropout: float=0.0,
                 lora_dtype: torch.dtype=None):
        """
        base: 要被包裹的线性层,可以是普通 nn.Linear,也可以是 bnb.nn.Linear4bit/8bitLt(QLoRA)
        lora_dtype: LoRA 自身参数的 dtype。None 时按优先级自动选:
          1) base.compute_dtype(bnb 量化层会暴露,通常 bf16)→ QLoRA 走这里
          2) base.weight.dtype 在 (uint8, int8) 时 → 兜底用 bf16(量化层 weight 是打包后的 uint8)
          3) 否则 → fp32(普通 LoRA 推荐:小量更新在 fp32 数值更稳)
        """
        super().__init__()
        self.base = base
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        self.scale = alpha / rank

        if lora_dtype is None:
            if hasattr(base, "compute_dtype") and base.compute_dtype is not None:
                lora_dtype = base.compute_dtype                    # QLoRA: 跟随 bnb 的 compute_dtype
            elif base.weight.dtype in (torch.uint8, torch.int8):
                lora_dtype = torch.bfloat16                         # 量化层但没暴露 compute_dtype,兜底
            else:
                lora_dtype = torch.float32                          # 普通 LoRA:强制 fp32 更稳

        device = base.weight.device
        self.lora_A = nn.Parameter(torch.empty((rank, base.in_features),
                                               dtype=lora_dtype, device=device))
        self.lora_B = nn.Parameter(torch.zeros((base.out_features, rank),
                                               dtype=lora_dtype, device=device))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.dropout_layer = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        base_out = self.base(x)
        # x → LoRA 精度算 → 算完降回 base_out 精度相加
        # QLoRA 场景:LoRA 和 base_out 都是 bf16,两次 cast 都是 no-op
        # 普通 LoRA(fp32 LoRA + bf16 base):前半在 fp32 算稳一些,最后降回 bf16
        x_lora = self.dropout_layer(x).to(self.lora_A.dtype)
        lora_out = (x_lora @ self.lora_A.T @ self.lora_B.T) * self.scale
        return base_out + lora_out.to(base_out.dtype)
    
def apply_lora(model: nn.Module, target_suffixes:tuple | str,
               rank=8, alpha=16, dropout=0.0, lora_dtype: torch.dtype=None):
    orig_model = getattr(model, "_orig_mod", model)  # torch.compile 包装的模型在 .orig_mod 里
    raw_model = getattr(orig_model, "module", orig_model)  # DDP 包装的模型在 .module 里

    if isinstance(target_suffixes, str):
        target_suffixes = (target_suffixes,)

    for p in raw_model.parameters():
        p.requires_grad = False

    for name, parent in raw_model.named_modules():
        for cname, child in parent.named_children():
            # QLoRA 的 bnb.Linear4bit/8bitLt 都是 nn.Linear 的子类,isinstance 一并匹配
            if cname in target_suffixes and isinstance(child, nn.Linear):
                lora = LoRALinear(child, rank=rank, alpha=alpha,
                                  dropout=dropout, lora_dtype=lora_dtype)
                setattr(parent, cname, lora)

def save_lora(model: nn.Module, path: str):
    orig_model = getattr(model, "_orig_mod", model)  # torch.compile 包装的模型在 .orig_mod 里
    raw_model = getattr(orig_model, "module", orig_model)  # DDP 包装的模型在 .module 里

    weights = {}
    meta = {"targets": set(), "rank": None, "alpha": None, "dropout": None}
    for name, module in raw_model.named_modules():
        if isinstance(module, LoRALinear):
            weights[name] = {
                'lora_A': module.lora_A.cpu(),
                'lora_B': module.lora_B.cpu(),
            }
            # 取模块名最后一段作为后缀,如 "layers.0.attention.Wq" -> "Wq"
            meta["targets"].add(name.rsplit(".", 1)[-1])
            # 假设全模型所有 LoRA 用同一套超参(常见做法)
            meta["rank"] = module.rank
            meta["alpha"] = module.alpha
            meta["dropout"] = module.dropout
    meta["targets"] = sorted(meta["targets"])

    torch.save({"meta": meta, "weights": weights}, path)

def load_lora(model: nn.Module, path: str):
    ckpt = torch.load(path, map_location="cpu")
    meta, weights = ckpt["meta"], ckpt["weights"]
    orig_model = getattr(model, "_orig_mod", model)  # torch.compile 包装的模型在 .orig_mod 里
    raw_model = getattr(orig_model, "module", orig_model)  # DDP 包装的模型在 .module 里

    # 如果模型还没挂 LoRA,根据 meta 自动 apply
    already_applied = any(isinstance(m, LoRALinear) for m in raw_model.modules())
    if not already_applied:
        apply_lora(
            raw_model,
            target_suffixes=tuple(meta["targets"]),
            rank=meta["rank"],
            alpha=meta["alpha"],
            dropout=meta["dropout"],
        )

    modules = dict(raw_model.named_modules())
    for name, state in weights.items():
        module = modules[name]
        if not isinstance(module, LoRALinear):
            raise RuntimeError(
                f"模块 {name} 不是 LoRALinear,ckpt 与当前模型结构不匹配"
            )
        module.lora_A.data.copy_(state['lora_A'].to(module.lora_A.device))
        module.lora_B.data.copy_(state['lora_B'].to(module.lora_B.device))

def save_lora_checkpoint(path, model, optimizer, scaler, lm_config,
                         epoch, step, global_step):
    """完整断点 checkpoint:LoRA 权重 + optimizer + scaler + 进度。
    签名兼容 train_utils.save_checkpoint,可作为 epoch_train(save_fn=...) 传入。

    注意:base 量化权重不存(假设 args.name_or_path 不变,加载时 from_pretrained 重新得到);
    lm_config 也不存(同上,base 自带 config)。这两个参数只是为了和 save_checkpoint 同签名。
    """
    raw_model = getattr(getattr(model, "_orig_mod", model), "module", model)

    # 复用 save_lora 的扫描逻辑
    weights = {}
    meta = {"targets": set(), "rank": None, "alpha": None, "dropout": None}
    for name, module in raw_model.named_modules():
        if isinstance(module, LoRALinear):
            weights[name] = {
                'lora_A': module.lora_A.cpu(),
                'lora_B': module.lora_B.cpu(),
            }
            meta["targets"].add(name.rsplit(".", 1)[-1])
            meta["rank"] = module.rank
            meta["alpha"] = module.alpha
            meta["dropout"] = module.dropout
    meta["targets"] = sorted(meta["targets"])

    torch.save({
        "meta": meta,
        "weights": weights,
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
        "step": step,
        "global_step": global_step,
    }, path)

def load_lora_checkpoint(model, optimizer, scaler, args, ckpt=None):
    """从 LoRA checkpoint 恢复 LoRA 权重 + optimizer + scaler + 进度。
    签名/语义模仿 train_utils.maybe_resume,返回 (start_epoch, start_step)。

    使用要求:
      - 模型必须已经 apply_lora 过(本函数不自动 apply,以便用户先按 CLI 超参 apply,
        再加载权重做严格对齐校验);
      - LoRA 超参(rank/alpha/target)必须与 ckpt 一致,否则形状不匹配会报错。
    """
    if ckpt is None:
        if getattr(args, "resume", None) is None or not os.path.isfile(args.resume):
            return 0, 0
        ckpt = torch.load(args.resume, map_location="cpu")

    raw_model = getattr(getattr(model, "_orig_mod", model), "module", model)
    modules = dict(raw_model.named_modules())

    # 灌 LoRA 权重
    for name, state in ckpt["weights"].items():
        m = modules.get(name)
        if m is None or not isinstance(m, LoRALinear):
            raise RuntimeError(
                f"模块 {name} 不是 LoRALinear,ckpt 与当前模型结构不匹配"
            )
        m.lora_A.data.copy_(state['lora_A'].to(m.lora_A.device))
        m.lora_B.data.copy_(state['lora_B'].to(m.lora_B.device))

    # 恢复 optimizer / scaler / 进度
    optimizer.load_state_dict(ckpt["optimizer"])
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])

    start_epoch = ckpt["epoch"]
    start_step = ckpt["step"]
    return start_epoch, start_step

def merge_lora(model: nn.Module, lora_path:str, save_path:str):
    """把 LoRA 折叠回 base.weight,保存为标准 HF 模型(不再依赖 LoRALinear 类)。

    ⚠️ 不支持量化 base(QLoRA):bnb.Linear4bit/8bitLt 的 weight 是打包的 uint8,
    直接 add(bf16 delta) 会破坏量化结构。QLoRA 部署请保留 base + adapter 分离,
    或自行先用 bitsandbytes.functional.dequantize_4bit 反量化再合并。
    """
    load_lora(model, lora_path)
    orig_model = getattr(model, "_orig_mod", model)  # torch.compile 包装的模型在 .orig_mod 里
    raw_model = getattr(orig_model, "module", orig_model)  # DDP 包装的模型在 .module 里

    for name, parent in raw_model.named_modules():
        for cname, child in list(parent.named_children()):
            if isinstance(child, LoRALinear):
                base = child.base
                # 量化层防呆:有 compute_dtype 属性(bnb 量化层会暴露),
                # 或者 weight 是 uint8/int8(打包后的量化权重),都拒绝 merge
                if hasattr(base, "compute_dtype") or base.weight.dtype in (torch.uint8, torch.int8):
                    raise RuntimeError(
                        f"模块 {name}.{cname} 的 base 是量化层"
                        f"({type(base).__name__}, weight.dtype={base.weight.dtype}),"
                        f"不支持直接 merge。QLoRA 部署请保留 base+adapter 分离加载。"
                    )
                delta = (child.lora_B @ child.lora_A) * child.scale
                # 保险起见 cast 到 base 的 dtype(LoRA 可能是 fp32,base 是 bf16)
                base.weight.data.add_(delta.to(base.weight.dtype))
                setattr(parent, cname, base)  # 把 LoRALinear 替换回原来的 Linear

    raw_model.save_pretrained(save_path)