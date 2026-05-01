from math import sqrt
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from transformers import PretrainedConfig
from urllib3 import Retry

class MyModelConfig(PretrainedConfig):
    model_type = "my_model"

    def __init__(
        self,
        dim: int = 768,                 #the dimension of the model
        n_layers: int = 12,             #the layers of the Transformer
        n_heads: int = 16,              #the heads of the multi-head attention
        n_kv_heads: int = 8,            #the heads of the key-value attention
        vocab_size: int = 6144,         #the vocabulary size
        hidden_dim: int = None,         #the hidden dimension of the MLP
        multiple_of: int = 64,          #
        norm_eps: float = 1e-5,         #the epsilon for layer normalization
        max_seq_len: int = 512,         #the maximum sequence length
        dropout: float = 0.0,           #the dropout rate
        flash_attention: bool = False,  #whether to use flash attention
        **kwargs
    ):
        self.dim = dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.multiple_of = multiple_of
        self.norm_eps = norm_eps
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.flash_attention = flash_attention

        super().__init__(**kwargs)
        
        
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(dim))#Parameter is a kind of Tensor, that is to be considered a module parameter.
        
    def _norm(self, x: torch.Tensor):
        #rsqrt is the hardware-friendly version of 1/sqrt(x). It is more efficient to compute and can be used for normalization purposes. In this case, it is used to normalize the input tensor x by its mean and variance.
        #keepdim=True means that the output will have the same number of dimensions as the input, and the reduced dimension will be of size 1. This is useful for broadcasting the result back to the original shape of x.
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    
    def forward(self, x: torch.Tensor):
        output = self._norm(x.float()).type_as(x)#type_as(x) means that the output will have the same data type as x. This is useful for ensuring that the output has the same precision as the input, especially when x is a half-precision tensor.
        return output * self.weight
    


def precompute_freqs_cs(seq_len: int, dim: int, theta: float = 10000.0):
    """
    这个函数是用来预计算位置编码的频率的。
    位置编码是一种用于Transformer模型中的技术，用于为输入序列中的每个位置提供唯一的表示。
    这个函数通过计算不同频率的正弦和余弦函数来生成位置编码的频率矩阵。
    但是和LLaMA官方计算不同的是，这里对每个值计算了两遍
    
    参数：
    seq_len: 输入序列的长度。
    dim: 模型的维度。这里是每个头的维度，而不是整个模型的维度。
    theta: 频率的基数，默认值为10000.0。
    
    输出：
    freqs_cos: 位置编码的余弦频率矩阵。形状为（seq_len, dim）。
    freqs_sin: 位置编码的正弦频率矩阵。形状为（seq_len, dim）。
    """
    
    freqs = 1.0 / theta ** (2 * (torch.arange(dim) // 2).float() / dim)
    idx = torch.arange(seq_len, device=freqs.device)
    freqs = torch.outer(idx, freqs).float()
    freqs_cos = torch.cos(freqs)
    freqs_sin = torch.sin(freqs)
    return freqs_cos, freqs_sin
    

def reshape_for_broadcast(freqs: torch.Tensor, X: torch.Tensor):
    """
    这个函数是用来将预计算的频率矩阵调整为适合与输入张量X进行广播的形状。
    参数：
    freqs: 预计算的频率矩阵，形状为（seq_len, head_dim）。
    X: 输入张量，形状为（batch_size, seq_len, n_heads, head_dim）。
    
    输出：
    freqs: 调整后的频率矩阵，形状为（1, seq_len, 1, head_dim）。
    """
    ndim = X.ndim
    assert 0 <= 1 < ndim
    assert freqs.shape == (X.shape[1], X.shape[-1])
    shape = [X.shape[i] if i == 1 or i == 3 else 1 for i in range(ndim)]
    return freqs.view(*shape)


def apply_rotary_positional_embedding(Xq:torch.Tensor,
                                      Xk: torch.Tensor,
                                      freqs_cos:torch.Tensor,
                                      freqs_sin:torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    这个函数是用来将旋转位置编码应用于查询张量Xq和键张量Xk的。
    RoPE最好在FP32精度下计算，因为它涉及到正弦和余弦函数的计算，这些函数在低精度下可能会导致数值不稳定。
    参数：
    Xq: 查询张量，形状为（batch_size, seq_len, n_heads, head_dim）。
    Xk: 键张量，形状为（batch_size, seq_len, n_heads, head_dim）。
    freqs_cos: 位置编码的余弦频率矩阵，形状为（seq_len, head_dim）。
    freqs_sin: 位置编码的正弦频率矩阵，形状为（seq_len, head_dim）。

    输出：
    Xq_out: 应用旋转位置编码后的查询张量，形状为（batch_size, seq_len, n_heads, head_dim）。
    Xk_out: 应用旋转位置编码后的键张量，形状为（batch_size, seq_len, n_heads, head_dim）。
    """
    original_dtype = Xq.dtype
    
    Xq = Xq.float()
    Xk = Xk.float()
    
    q_pair = Xq.reshape(*Xq.shape[:-1], -1, 2)
    q_trans = torch.stack([-q_pair[..., 1], q_pair[..., 0]], dim=-1)
    Xq_trans = q_trans.reshape(*Xq.shape)
    k_pair = Xk.reshape(Xk.shape[:-1] + (-1, 2))
    k_trans = torch.stack([-k_pair[..., 1], k_pair[..., 0]], dim=-1)
    Xk_trans = k_trans.reshape(*Xk.shape)
    
    freqs_cos = reshape_for_broadcast(freqs_cos, Xq)
    freqs_sin = reshape_for_broadcast(freqs_sin, Xq)
    
    Xq_out = Xq * freqs_cos + Xq_trans * freqs_sin
    Xk_out = Xk * freqs_cos + Xk_trans * freqs_sin
    return Xq_out.type_as(original_dtype), Xk_out.type_as(original_dtype)


def repeat_kv(x:torch.Tensor, n_rep: int) -> torch.Tensor:
    bs, seq_len, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, seq_len, n_kv_heads, n_rep, head_dim)
        .reshape(bs, seq_len, n_kv_heads * n_rep, head_dim)
    )
    
class Attention(nn.Module):
    def __init__(self, args: MyModelConfig):
        super().__init__()
        self.dim = args.dim
        self.n_heads = args.n_heads
        self.head_dim = self.dim // self.n_heads           
        if args.n_kv_heads is None:
            self.n_kv_heads = args.n_heads
        else:
            self.n_kv_heads = args.n_kv_heads
        assert self.n_heads % self.n_kv_heads == 0
        self.n_reps = self.n_heads // self.n_kv_heads
        self.dropout = args.dropout
        
        """这里的代码是用来设置模型并行的相关参数的。但是在该处为无效设置，
        因为需要使用fairscale或者torch.distributed来实现模型并行，并且需要在训练脚本中进行相应的设置和调用。
        后续计划添加相关设置"""      
        self.model_parallel_size = 1
        self.n_local_heads = self.n_heads // self.model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // self.model_parallel_size
        
        self.Wq = nn.Linear(self.dim, self.n_heads * self.head_dim, bias=False)
        self.Wk = nn.Linear(self.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.Wv = nn.Linear(self.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.Wo = nn.Linear(self.n_heads * self.head_dim, self.dim, bias=False)
        
        self.attn_dropout = nn.Dropout(self.dropout)
        self.resi_dropout = nn.Dropout(self.dropout)
        
        self.flash = hasattr(F, "scaled_dot_product_attention")        
        
    def forward(self, 
                X: torch.Tensor,
                freqs_cos: torch.Tensor,
                freqs_sin: torch.Tensor,
                key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        这个函数是用来执行注意力机制的前向传播的。
        参数：
        X: 输入张量，形状为（batch_size, seq_len, dim）。
        freqs_cos: 位置编码的余弦频率矩阵，形状为（seq_len, head_dim）。
        freqs_sin: 位置编码的正弦频率矩阵，形状为（seq_len, head_dim）。
        key_padding_mask: 可选的键填充掩码，形状为（batch_size, seq_len）。是为了在多批次训练时对齐每个序列的长度。
        输出：
        output: 注意力机制的输出张量，形状为（batch_size, seq_len, dim）。
        """  
        bs, seq_len, _ = X.shape
        
        Xq, Xk, Xv = self.Wq(X), self.Wk(X), self.Wv(X)
        
        Xq = Xq.view(bs, seq_len, self.n_local_heads, self.head_dim)
        Xk = Xk.view(bs, seq_len, self.n_local_kv_heads, self.head_dim)
        Xv = Xv.view(bs, seq_len, self.n_local_kv_heads, self.head_dim)
        
        Xq, Xk = apply_rotary_positional_embedding(Xq, Xk, freqs_cos, freqs_sin)
        
        Xk = repeat_kv(Xk, self.n_reps)
        Xv = repeat_kv(Xv, self.n_reps)
        
        Xq = Xq.transpose(1, 2)
        Xk = Xk.transpose(1, 2)
        Xv = Xv.transpose(1, 2)
        
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask[:, None, None, :].to(dtype=torch.bool)
        
        if self.flash:
            if key_padding_mask is not None:
                casual_mask = torch.full((1, 1, seq_len, seq_len), 1, dtype=torch.bool, device=key_padding_mask.device).tril()
                attn_mask = key_padding_mask & casual_mask
                attention = F.scaled_dot_product_attention(
                    Xq,
                    Xk,
                    Xv,
                    attn_mask=attn_mask,
                    dropout_p=self.dropout if self.training else 0,
                    is_causal=False,
                )
            else:
                attention = F.scaled_dot_product_attention(
                    Xq,
                    Xk,
                    Xv,
                    dropout_p=self.dropout if self.training else 0,
                    is_causal=True
                )
        else:
            scores = torch.matmul(Xq, Xk.transpose(2, 3)) / sqrt(self.head_dim)
            if key_padding_mask is not None:
                scores = scores.masked_fill(~key_padding_mask, float('-inf'))
            casual_mask = torch.full((1, 1, seq_len, seq_len), float('-inf'), device=Xq.device).triu(diagonal=1)
            scores = scores + casual_mask
            scores = torch.softmax(scores.float(), dim=-1).type_as(scores)
            attention = self.attn_dropout(scores).matmul(Xv)
        
        attention = attention.transpose(1, 2).contiguous().view(bs, seq_len, -1)
        
        output = self.resi_dropout(attention)
        output = self.Wo(output)
        return output


class FeedForward(nn.Module):
    """
    这个类是用来实现Transformer中的前馈神经网络（Feed-Forward Network）的。
    这里的forward有三个线性层，分别是up、gate和down。up和gate层将输入张量X映射到hidden_dim维度的空间中，而down层将hidden_dim维度的张量映射回dim维度的空间中。
    在前馈神经网络中，通常会使用一个激活函数来增加模型的非线性能力。在这个实现中，使用了SiLU（Sigmoid Linear Unit）作为激活函数。
    """
    def __init__(self, dim:int, hidden_dim:int, mutiple_of:int, dropout:float):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = dim * 4
            hidden_dim = 2 * hidden_dim // 3
            hidden_dim = mutiple_of * ((hidden_dim + mutiple_of - 1) // mutiple_of) 

        self.up = nn.Linear(dim, hidden_dim, bias=False)
        self.gate = nn.Linear(dim, hidden_dim, bias=False)
        self.down = nn.Linear(hidden_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, X:torch.Tensor):
        return self.dropout(self.down(F.silu(self.gate(X)) * self.up(X)))
        
class TransformerBlock(nn.Module):
    """
    这个类是用来实现Transformer中的一个块（Transformer Block）的。
    一个Transformer块通常包含一个多头自注意力机制和一个前馈神经网络。在这个实现中，Transformer块包含一个RMSNorm层、一个Attention层、另一个RMSNorm层和一个FeedForward层。
    这里在attention和FFN之间使用了残差连接（residual connection），即将输入张量X与注意力机制的输出相加，得到h，然后将h与前馈神经网络的输出相加，得到最终的输出。
    """
    def __init__(self, block_num:int, args:MyModelConfig):
        super().__init__()     
        self.block_num = block_num
        
        self.attn_RMSNorm = RMSNorm(args.dim, args.norm_eps)
        self.attention = Attention(args)
        self.FFN_RMSNorm = RMSNorm(args.dim, args.norm_eps)
        self.FFN = FeedForward(args.dim, args.hidden_dim, args.multiple_of, args.dropout)
    
    def forward(self, 
                X:torch.Tensor, 
                freqs_cos:torch.Tensor, 
                freqs_sin:torch.Tensor, 
                key_padding_mask:torch.Tensor
        ):
        h = X + self.attention(self.attn_RMSNorm(X), freqs_cos, freqs_sin, key_padding_mask)
        output = h + self.FFN(self.FFN_RMSNorm(h))
        return output
    
    
    
if __name__ == "__main__":
    
    """这个代码块是用来测试MyModelConfig和RMSNorm类的功能的。"""
    # config = MyModelConfig()
    # print(config)
    # norm = RMSNorm(config.dim, config.norm_eps)
    # x = torch.randn(2, 3, config.dim)
    # output = norm(x)
    # print(output.shape)
    
    """这个代码块是用来测试precompute_freqs_cs和apply_rotary_positional_embedding函数的功能的。"""
    # Xq = torch.randn(1, 2, 1, 4)
    # Xk = torch.randn(1, 2, 1, 4)
    # freqs_cos, freqs_sin = precompute_freqs_cs(2, 4)
    # print(Xq)
    # print(Xk)
    # print(freqs_cos.shape)
    # print(freqs_sin.shape)
    # Xq_out, Xk_out = apply_rotary_positional_embedding(Xq, Xk, freqs_cos, freqs_sin)
    # print(Xq_out.shape)
    # print(Xk_out.shape)
    # print(Xq_out)
    # print(Xk_out)
    
    """这个代码块是用来测试Attention类的功能的。"""
    # args = MyModelConfig()
    # attention_model = Attention(args)
    
    # bs = 1
    # seq_len = 50
    # X = torch.randn(bs, seq_len, args.dim)
    
    # freqs_cos, freqs_sin = precompute_freqs_cs(seq_len, args.dim // args.n_heads)
    
    # output = attention_model(X, freqs_cos, freqs_sin)
    
    # print("output shape", output.shape)
    
    """这个代码块是用来测试FeedForward类的功能的。"""
    # args = MyModelConfig()
    # feedforward_model = FeedForward(args.dim, args.hidden_dim, args.multiple_of, args.dropout)
    # bs = 1
    # seq_len = 50
    # X = torch.randn(bs, seq_len, args.dim)
    # output = feedforward_model(X)
    # print("output shape", output.shape)