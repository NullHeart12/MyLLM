import argparse
import json
import os
import random
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.trainer_pt_utils import LengthGroupedSampler

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deal_dataset.dataset import DPODataset, DPOCollator
from dpo.dpo_loss import compute_dpo_loss, logits_to_log_probs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a DPO model on tokenized DPO pairs.")
    parser.add_argument("--policy_model_path", type=str, required=True,
                        help="DPO 后的 policy 模型目录，通常是 dpo 输出的 final_hf。")
    parser.add_argument("--ref_model_path", type=str, required=True,
                        help="DPO 前的 reference/base 模型目录。")
    parser.add_argument("--eval_data_path", type=str, required=True,
                        help="DPOProcessor 产出的 tokenized Arrow 目录或 json/jsonl 文件。")
    parser.add_argument("--out_dir", type=str, default=os.path.join(PROJECT_ROOT, "dpo", "eval", "eval_out"))

    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--device", type=str, default="cuda:0")
    return parser.parse_args()


def get_dtype(dtype: str) -> torch.dtype:
    if dtype == "bfloat16":
        return torch.bfloat16
    if dtype == "float16":
        return torch.float16
    return torch.float32


def maybe_subset(dataset: DPODataset, max_eval_samples: int | None):
    lengths = dataset.get_len()
    if max_eval_samples is None or max_eval_samples <= 0 or max_eval_samples >= len(dataset):
        return dataset, lengths
    rng = random.Random(42)
    indices = rng.sample(range(len(dataset)), max_eval_samples)
    subset_lengths = [lengths[i] for i in indices]
    return Subset(dataset, indices), subset_lengths


def model_log_probs(model, input_ids, labels, attention_mask):
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
    )
    return logits_to_log_probs(outputs.logits, labels)


@torch.no_grad()
def evaluate(
    policy_model,
    ref_model,
    data_loader: DataLoader,
    device: torch.device,
    beta: float,
) -> dict[str, Any]:
    policy_model.eval()
    ref_model.eval()

    total_samples = 0
    total_loss = 0.0
    total_reward_margin = 0.0
    total_preference_correct = 0
    total_policy_preference_correct = 0
    total_ref_preference_correct = 0
    total_policy_logp_margin = 0.0
    total_ref_logp_margin = 0.0
    total_chosen_reward = 0.0
    total_rejected_reward = 0.0

    for batch in tqdm(data_loader, desc="DPO eval"):
        chosen_ids = batch["chosen_ids"].to(device)
        chosen_labels = batch["chosen_labels"].to(device)
        chosen_attention_mask = batch["chosen_attention_mask"].to(device)
        rejected_ids = batch["rejected_ids"].to(device)
        rejected_labels = batch["rejected_labels"].to(device)
        rejected_attention_mask = batch["rejected_attention_mask"].to(device)

        policy_chosen_log_probs = model_log_probs(
            policy_model, chosen_ids, chosen_labels, chosen_attention_mask
        )
        policy_rejected_log_probs = model_log_probs(
            policy_model, rejected_ids, rejected_labels, rejected_attention_mask
        )
        ref_chosen_log_probs = model_log_probs(
            ref_model, chosen_ids, chosen_labels, chosen_attention_mask
        )
        ref_rejected_log_probs = model_log_probs(
            ref_model, rejected_ids, rejected_labels, rejected_attention_mask
        )

        loss = compute_dpo_loss(
            ref_chosen_log_probs=ref_chosen_log_probs,
            ref_rejected_log_probs=ref_rejected_log_probs,
            policy_chosen_log_probs=policy_chosen_log_probs,
            policy_rejected_log_probs=policy_rejected_log_probs,
            beta=beta,
        )

        policy_chosen_seq = policy_chosen_log_probs.sum(dim=1)
        policy_rejected_seq = policy_rejected_log_probs.sum(dim=1)
        ref_chosen_seq = ref_chosen_log_probs.sum(dim=1)
        ref_rejected_seq = ref_rejected_log_probs.sum(dim=1)
        policy_logp_margin = policy_chosen_seq - policy_rejected_seq
        ref_logp_margin = ref_chosen_seq - ref_rejected_seq

        chosen_rewards = beta * (policy_chosen_seq - ref_chosen_seq)
        rejected_rewards = beta * (policy_rejected_seq - ref_rejected_seq)
        reward_margin = chosen_rewards - rejected_rewards

        batch_size = chosen_ids.size(0)
        total_samples += batch_size
        total_loss += loss.item() * batch_size
        total_reward_margin += reward_margin.sum().item()
        total_preference_correct += (reward_margin > 0).sum().item()
        total_policy_preference_correct += (policy_logp_margin > 0).sum().item()
        total_ref_preference_correct += (ref_logp_margin > 0).sum().item()
        total_policy_logp_margin += policy_logp_margin.sum().item()
        total_ref_logp_margin += ref_logp_margin.sum().item()
        total_chosen_reward += chosen_rewards.sum().item()
        total_rejected_reward += rejected_rewards.sum().item()

    total_samples = max(1, total_samples)
    ref_preference_acc = total_ref_preference_correct / total_samples
    policy_preference_acc = total_policy_preference_correct / total_samples
    ref_logp_margin = total_ref_logp_margin / total_samples
    policy_logp_margin = total_policy_logp_margin / total_samples
    return {
        "eval/dpo_loss": total_loss / total_samples,
        "eval/preference_acc": total_preference_correct / total_samples,
        "eval/reward_margin": total_reward_margin / total_samples,
        "eval/ref_preference_acc": ref_preference_acc,
        "eval/policy_preference_acc": policy_preference_acc,
        "eval/preference_acc_delta": policy_preference_acc - ref_preference_acc,
        "eval/ref_logp_margin": ref_logp_margin,
        "eval/policy_logp_margin": policy_logp_margin,
        "eval/logp_margin_delta": policy_logp_margin - ref_logp_margin,
        "eval/chosen_reward": total_chosen_reward / total_samples,
        "eval/rejected_reward": total_rejected_reward / total_samples,
        "eval/n_samples": total_samples,
    }


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device(args.device)
    dtype = get_dtype(args.dtype)

    tokenizer = AutoTokenizer.from_pretrained(args.policy_model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading policy model: {args.policy_model_path}")
    policy_model = AutoModelForCausalLM.from_pretrained(
        args.policy_model_path,
        torch_dtype=dtype,
    ).to(device)
    policy_model.config.use_cache = False

    print(f"Loading reference model: {args.ref_model_path}")
    ref_model = AutoModelForCausalLM.from_pretrained(
        args.ref_model_path,
        torch_dtype=dtype,
    ).to(device)
    ref_model.config.use_cache = False
    for param in ref_model.parameters():
        param.requires_grad_(False)

    eval_dataset = DPODataset(args.eval_data_path)
    eval_dataset, lengths = maybe_subset(eval_dataset, args.max_eval_samples)

    generator = torch.Generator()
    generator.manual_seed(42)
    sampler = LengthGroupedSampler(
        dataset=eval_dataset,
        batch_size=args.batch_size,
        lengths=lengths,
        generator=generator,
    )

    data_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=False,
        collate_fn=DPOCollator(tokenizer.pad_token_id),
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    metrics = evaluate(
        policy_model=policy_model,
        ref_model=ref_model,
        data_loader=data_loader,
        device=device,
        beta=args.beta,
    )

    result = {
        "policy_model_path": args.policy_model_path,
        "ref_model_path": args.ref_model_path,
        "eval_data_path": args.eval_data_path,
        "beta": args.beta,
        "group_by_length": True,
        "metrics": metrics,
    }
    metrics_path = os.path.join(args.out_dir, "dpo_eval_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("\n===== DPO Eval Metrics =====")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")
    print(f"Saved metrics to: {metrics_path}")


if __name__ == "__main__":
    main()
