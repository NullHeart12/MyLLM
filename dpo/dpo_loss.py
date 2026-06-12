import torch
import torch.nn.functional as F

def logits_to_log_probs(
    logits:torch.Tensor, 
    labels:torch.Tensor
)->torch.Tensor:
    probs = F.log_softmax(logits, dim=-1)
    loss_mask = labels != -100
    safe_labels = labels.masked_fill(~loss_mask, 0)
    log_probs_per_token = torch.gather(
        probs,
        dim=-1,
        index=safe_labels.unsqueeze(-1),
    ).squeeze(-1)
    return log_probs_per_token * loss_mask

def compute_dpo_loss(
    ref_chosen_log_probs:torch.Tensor,
    ref_rejected_log_probs:torch.Tensor,
    policy_chosen_log_probs:torch.Tensor,
    policy_rejected_log_probs:torch.Tensor,
    beta:float
) -> torch.Tensor:
    ref_chosen_log_probs = ref_chosen_log_probs.sum(dim=1)
    ref_rejected_log_probs = ref_rejected_log_probs.sum(dim=1)
    policy_chosen_log_probs = policy_chosen_log_probs.sum(dim=1)
    policy_rejected_log_probs = policy_rejected_log_probs.sum(dim=1)
    
    ref_diff = ref_chosen_log_probs - ref_rejected_log_probs
    policy_diff = policy_chosen_log_probs - policy_rejected_log_probs
    loss = -F.logsigmoid(beta * (policy_diff - ref_diff))
    return loss.mean()
