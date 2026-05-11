import os
import argparse

from model import MyModelConfig, Transformer

from transformers import AutoTokenizer
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def set_ddp():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def destroy_ddp():
    dist.destroy_process_group()
    
def logger(content:str):
    print(content)

def load_tokenizer(tokenizer_path:str):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    return tokenizer

def load_model(
    lm_config:MyModelConfig,
    args:argparse.Namespace
) -> DDP:
    
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    my_model = Transformer(lm_config)
    my_model = my_model.to(lm_config.device)
    my_model = DDP(
        my_model,
        device_ids=[args.local_rank],
        output_device=args.local_rank,
    )
    
    if args.is_main:
        logger(f"模型参数数量: {count_parameters(my_model) / 1e6:.3f}M")
    
    return my_model
