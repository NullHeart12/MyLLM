from ast import arg
import select
from tkinter import NO
from turtle import forward

from regex import F
import torch
from torch import nn

from transformers import PretrainedConfig

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
    
    freqs = 1.0 / theta ** (2 * (torch.arange(dim) // 2) / dim)
    idx = torch.arange(seq_len, device=freqs.device)
    freqs = torch.outer(idx, freqs)
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
    参数：
    Xq: 查询张量，形状为（batch_size, seq_len, n_heads, head_dim）。
    Xk: 键张量，形状为（batch_size, seq_len, n_heads, head_dim）。
    freqs_cos: 位置编码的余弦频率矩阵，形状为（seq_len, head_dim）。
    freqs_sin: 位置编码的正弦频率矩阵，形状为（seq_len, head_dim）。

    输出：
    Xq_out: 应用旋转位置编码后的查询张量，形状为（batch_size, seq_len, n_heads, head_dim）。
    Xk_out: 应用旋转位置编码后的键张量，形状为（batch_size, seq_len, n_heads, head_dim）。
    """
       
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
    return Xq_out, Xk_out 
    
    
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
        self.dim = args.dim
        self.head_dim = self.dim // self.n_heads    
        self.n_heads = args.n_heads
        if args.n_kv_heads is None:
            self.n_kv_heads = args.n_heads
        else:
            self.n_kv_heads = args.n_kv_heads
        assert self.n_heads % self.n_kv_heads == 0
        
        self.model_parallel_size = 1
        self.n_local_heads = self.n_heads // self.model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // self.model_parallel_size
        
        self.Wq = nn.Linear(self.dim, self.n_heads * self.head_dim, bias=False)
        self.Wk = nn.Linear(self.dim, self.n_heads * self.head_dim, bias=False)
        self.Wv = nn.Linear(self.dim, self.n_heads * self.head_dim, bias=False)
        self.Wo = nn.Linear(self.n_heads * self.head_dim, self.dim, bias=False)
        
        
        
    def forward(self, 
                X: torch.Tensor,
                freqs_cos: torch.Tensor,
                freqs_sin: torch.Tensor,
                attention_mask: torch.Tensor
    ) -> torch.Tensor:
        bs, seq_len, _ = X.shape
        
        Xq, Xk, Xv = self.Wq(X), self.Wk(X), self.Wv(X)
        
        Xq = Xq.view(bs, seq_len, self.n_local_heads, self.head_dim)
        Xk = Xk.view(bs, seq_len, self.n_local_kv_heads, self.head_dim)
        Xv = Xv.view(bs, seq_len, self.n_local_kv_heads, self.head_dim)
        
        Xq, Xk = apply_rotary_positional_embedding(Xq, Xk, freqs_cos, freqs_sin)
        
        
        
        pass
        

    
    
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