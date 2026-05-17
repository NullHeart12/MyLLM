from math import sqrt
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

class MyModelConfig(PretrainedConfig):
    model_type = "my_model"

    def __init__(
        self,
        dim: int = 1024,                 #the dimension of the model
        n_layers: int = 28,             #the layers of the Transformer
        n_heads: int = 16,              #the heads of the multi-head attention
        n_kv_heads: int = 4,            #the heads of the key-value attention
        vocab_size: int = 25600,        #the vocabulary size
        hidden_dim: int = None,         #the hidden dimension of the MLP
        multiple_of: int = 64,          #
        use_moe: bool = False,
        n_experts: int = 8,
        moe_top_k: int = 3,
        router_aux_loss_coef: float = 1e-2,
        norm_eps: float = 1e-5,         #the epsilon for layer normalization
        max_seq_len: int = 1024,        #the maximum sequence length
        dropout: float = 0.0,           #the dropout rate
        flash_attention: bool = False,  #whether to use flash attention
        bos_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
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
        self.use_moe = use_moe
        self.n_experts = n_experts
        self.moe_top_k = moe_top_k
        self.router_aux_loss_coef = router_aux_loss_coef
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id

        # ---- 基础维度 ----
        assert self.dim > 0, "dim must be positive"
        assert self.n_layers > 0, "n_layers must be positive"
        assert self.vocab_size > 0, "vocab_size must be positive"
        assert self.max_seq_len > 0, "max_seq_len must be positive"
        assert self.multiple_of > 0, "multiple_of must be positive"

        # ---- 注意力相关 ----
        assert self.n_heads > 0, "n_heads must be positive"
        assert self.n_kv_heads is None or self.n_kv_heads > 0, "n_kv_heads must be positive"
        assert self.dim % self.n_heads == 0, \
            f"dim ({self.dim}) must be divisible by n_heads ({self.n_heads})"
        if self.n_kv_heads is not None:
            assert self.n_heads % self.n_kv_heads == 0, \
                f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})"
            assert self.n_kv_heads <= self.n_heads, \
                "n_kv_heads must not exceed n_heads"

        # ---- FFN 相关 ----
        if self.hidden_dim is not None:
            assert self.hidden_dim > 0, "hidden_dim must be positive"
            assert self.hidden_dim % self.multiple_of == 0, \
                f"hidden_dim ({self.hidden_dim}) must be a multiple of multiple_of ({self.multiple_of})"

        # ---- 正则/数值 ----
        assert 0.0 <= self.dropout < 1.0, "dropout must be in [0, 1)"
        assert self.norm_eps > 0, "norm_eps must be positive"

        # ---- MoE 相关 ----
        if self.use_moe:
            assert self.n_experts > 0, "n_experts must be positive when use_moe=True"
            assert 1 <= self.moe_top_k <= self.n_experts, \
                f"moe_top_k ({self.moe_top_k}) must be in [1, n_experts={self.n_experts}]"
            assert self.router_aux_loss_coef >= 0, "router_aux_loss_coef must be non-negative"

        # ---- 特殊 token ----
        assert bos_token_id is not None, "bos_token_id must be specified"
        assert eos_token_id is not None, "eos_token_id must be specified"
        assert pad_token_id is not None, "pad_token_id must be specified"

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

    freqs = 1.0 / theta ** (2 * (torch.arange(dim, dtype=torch.float32) // 2) / dim)
    idx = torch.arange(seq_len, dtype=torch.float32)
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
    return Xq_out.to(original_dtype), Xk_out.to(original_dtype)


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

        self.Wq = nn.Linear(self.dim, self.n_heads * self.head_dim, bias=False)
        self.Wk = nn.Linear(self.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.Wv = nn.Linear(self.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.Wo = nn.Linear(self.n_heads * self.head_dim, self.dim, bias=False)

        self.attn_dropout = nn.Dropout(self.dropout)
        self.resid_dropout = nn.Dropout(self.dropout)

        self.flash = args.flash_attention and hasattr(F, "scaled_dot_product_attention")
        
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
        
        Xq = Xq.view(bs, seq_len, self.n_heads, self.head_dim)
        Xk = Xk.view(bs, seq_len, self.n_kv_heads, self.head_dim)
        Xv = Xv.view(bs, seq_len, self.n_kv_heads, self.head_dim)
        
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
                causal_mask = torch.full((1, 1, seq_len, seq_len), 1, dtype=torch.bool, device=key_padding_mask.device).tril()
                attn_mask = key_padding_mask & causal_mask
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
            causal_mask = torch.full((1, 1, seq_len, seq_len), float('-inf'), device=Xq.device).triu(diagonal=1)
            scores = scores + causal_mask
            scores = torch.softmax(scores.float(), dim=-1).type_as(scores)
            attention = self.attn_dropout(scores).matmul(Xv)

        attention = attention.transpose(1, 2).contiguous().view(bs, seq_len, -1)

        output = self.resid_dropout(attention)
        output = self.Wo(output)
        return output


class FeedForward(nn.Module):
    """
    这个类是用来实现Transformer中的前馈神经网络（Feed-Forward Network）的。
    这里的forward有三个线性层，分别是up、gate和down。up和gate层将输入张量X映射到hidden_dim维度的空间中，而down层将hidden_dim维度的张量映射回dim维度的空间中。
    在前馈神经网络中，通常会使用一个激活函数来增加模型的非线性能力。在这个实现中，使用了SiLU（Sigmoid Linear Unit）作为激活函数。
    """
    def __init__(self, dim:int, hidden_dim:int, multiple_of:int, dropout:float):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = dim * 4
            hidden_dim = 2 * hidden_dim // 3
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of) 

        self.up = nn.Linear(dim, hidden_dim, bias=False)
        self.gate = nn.Linear(dim, hidden_dim, bias=False)
        self.down = nn.Linear(hidden_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, X:torch.Tensor):
        return self.dropout(self.down(F.silu(self.gate(X)) * self.up(X)))
    
class MoEFeedForward(nn.Module):
    def __init__(
            self, 
            dim:int, 
            hidden_dim:int, 
            multiple_of:int, 
            dropout:float, 
            n_experts:int,
            top_k:int,
            router_aux_loss_coef:float,
        ):
        super().__init__()
        self.top_k = top_k
        self.router_aux_loss_coef = router_aux_loss_coef
        if hidden_dim is None:
            hidden_dim = dim * 4
            hidden_dim = 2 * hidden_dim // 3
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of) 

        self.n_experts = n_experts
        self.gate = nn.Linear(dim, n_experts, bias=False)
        self.experts = nn.ModuleList([
            FeedForward(dim, hidden_dim, multiple_of, dropout)
            for _ in range(n_experts)
        ])
    
    # old version        
    # def forward(self, x:torch.Tensor):
    #     bs, seq_len, dim = x.shape
    #     x_flaten = x.view(-1, dim)
    #     scores = F.softmax(self.gate(x_flaten), dim=-1)
    #     topk_weights, topk_ids = torch.topk(scores, k=self.top_k, dim=-1)
    #     topk_weights = topk_weights / torch.sum(topk_weights, dim=-1, keepdim=True)
        
    #     y = x.new_zeros(bs * seq_len, dim)
    #     for i, expert in enumerate(self.experts):
    #         chosen = (topk_ids == i)
    #         if chosen.any():
    #             tokens = chosen.any(-1).nonzero().squeeze(1)
    #             weight_for_i = topk_weights[chosen].sum(-1)
    #             weights = weight_for_i[tokens].view(-1, 1)
    #             y[tokens] += weights * expert(x_flaten[tokens])
    #         elif self.training:
    #             y[0, 0] += 0 * sum(p.sum() for p in expert.parameters())
        
    #     if self.training and self.router_aux_loss_coef > 0:
    #         one_hot = F.one_hot(topk_ids, num_classes=self.n_experts).float()
    #         tokens_per_expert = one_hot.sum(dim=(0, 1))
    #         f = tokens_per_expert / (bs * seq_len * self.top_k)
                
    #         P = scores.mean(dim=0)
            
    #         self.loss = self.n_experts * (f * P).sum() * self.router_aux_loss_coef
    #     else:
            # self.loss = x.new_tensor(0.0)   # 或 torch.tensor(0.0, device=x.device)
        
    #     return y.view(bs, seq_len, dim) 
    
    def forward(self, x:torch.Tensor):
        bs, seq_len, dim = x.shape
        x_flat = x.view(-1, dim)
        scores = F.softmax(self.gate(x_flat), dim=-1)
        
        topk_expert_weights, topk_expert_ids = torch.topk(
            scores, k=self.top_k, dim=-1
        )
        topk_expert_weights = topk_expert_weights / topk_expert_weights.sum(dim=-1, keepdim=True)
        
        flat_expert_ids = topk_expert_ids.view(-1)
        flat_weights = topk_expert_weights.view(-1)
        flat_token_ids = torch.arange(bs * seq_len, device=x.device).repeat_interleave(self.top_k)
        
        sorted_expert_ids, indices = torch.sort(flat_expert_ids)
        sorted_weights = flat_weights[indices]
        sorted_token_ids = flat_token_ids[indices]
        
        tokens_per_expert = torch.bincount(sorted_expert_ids, minlength=self.n_experts)
        boundaries = tokens_per_expert.cumsum(0).tolist()
        
        y = x.new_zeros(bs * seq_len, dim)
        prev = 0
        for i, expert in enumerate(self.experts):
            cur = boundaries[i]
            if cur == prev:
                if self.training:
                    y[0, 0] += 0 * sum(p.sum() for p in expert.parameters())
                continue
            token_idx = sorted_token_ids[prev:cur]
            weights = sorted_weights[prev:cur].view(-1, 1)
            out_i = weights * expert(x_flat[token_idx])
            y[token_idx] += out_i
            prev = cur
            
        if self.training and self.router_aux_loss_coef > 0:
            one_hot = F.one_hot(topk_expert_ids, num_classes=self.n_experts).float()
            tokens_per_expert = one_hot.sum(dim=(0, 1))
            f = tokens_per_expert / (bs * seq_len * self.top_k)
                
            P = scores.mean(dim=0)
            
            self.loss = self.n_experts * (f * P).sum() * self.router_aux_loss_coef
        else:
            self.loss = x.new_tensor(0.0)   # 或 torch.tensor(0.0, device=x.device)
            
        return y.view(bs, seq_len, dim)

        
class TransformerBlock(nn.Module):
    """
    这个类是用来实现Transformer中的一个块（Transformer Block）的。
    一个Transformer块通常包含一个多头自注意力机制和一个前馈神经网络。在这个实现中，Transformer块包含一个RMSNorm层、一个Attention层、另一个RMSNorm层和一个FeedForward层。
    这里在attention和FFN之间使用了残差连接（residual connection），即将输入张量X与注意力机制的输出相加，得到h，然后将h与前馈神经网络的输出相加，得到最终的输出。
    """
    def __init__(self, block_id:int, args:MyModelConfig):
        super().__init__()     
        self.block_id = block_id

        self.attn_RMSNorm = RMSNorm(args.dim, args.norm_eps)
        self.attention = Attention(args)
        self.FFN_RMSNorm = RMSNorm(args.dim, args.norm_eps)
        if args.use_moe:
            self.FFN = MoEFeedForward(
                dim=args.dim,
                hidden_dim=args.hidden_dim,
                multiple_of=args.multiple_of,
                dropout=args.dropout,
                n_experts=args.n_experts,
                top_k=args.moe_top_k,
                router_aux_loss_coef=args.router_aux_loss_coef,
            )
        else:
            self.FFN = FeedForward(args.dim, args.hidden_dim, args.multiple_of, args.dropout)
    
    def forward(self, 
                X:torch.Tensor, 
                freqs_cos:torch.Tensor, 
                freqs_sin:torch.Tensor, 
                key_padding_mask:Optional[torch.Tensor]=None
        ):
        h = X + self.attention(self.attn_RMSNorm(X), freqs_cos, freqs_sin, key_padding_mask)
        output = h + self.FFN(self.FFN_RMSNorm(h))
        return output
    
    
class Transformer(PreTrainedModel):
    config_class = MyModelConfig
    
    def __init__(self, args:MyModelConfig):
        super().__init__(args)
        
        self.args = args
        self.vocab_size = args.vocab_size
        self.dim = args.dim
        
        self.token_embeddings = nn.Embedding(self.vocab_size, self.dim)
        
        self.dropout = nn.Dropout(args.dropout)
        
        self.n_layers = args.n_layers
        self.layers = nn.ModuleList()
        for id in range(self.n_layers):
            cur_layer = TransformerBlock(id, args)
            self.layers.append(cur_layer)
            
        self.norm_eps = args.norm_eps
        self.RMSNorm = RMSNorm(self.dim, self.norm_eps)
        
        self.output = nn.Linear(self.dim, self.vocab_size, bias=False)    
        
        self.token_embeddings.weight = self.output.weight
        
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("Wo.weight") or pn.endswith("down.weight"):
                nn.init.normal_(p, mean=0, std=0.02 / sqrt(2 * self.n_layers))
        
        self.max_seq_len = args.max_seq_len
        freqs_cos, freqs_sin = precompute_freqs_cs(
                                    self.max_seq_len * 2,
                                    self.dim // args.n_heads,
                                )
        self.register_buffer("freqs_cos", freqs_cos)
        self.register_buffer("freqs_sin", freqs_sin)
        
        self.eos_token_id = args.eos_token_id
        self.pad_token_id = args.pad_token_id
        
    def _init_weights(self, module:nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0, std=0.02)
            
    def _prepare_padding_mask(self,
                              key_padding_mask:Optional[torch.Tensor],
                              tokens:Optional[torch.Tensor] 
        ) -> Optional[torch.Tensor]:
        if key_padding_mask is None:
            return None
        if key_padding_mask.dim() == 4:
            key_padding_mask = key_padding_mask[:, 0, 0, :]
        elif key_padding_mask.dim() == 3:
            key_padding_mask = key_padding_mask[:, 0, :]
        key_padding_mask = key_padding_mask.to(tokens.device)
        if key_padding_mask.dtype != torch.bool:
            key_padding_mask = key_padding_mask > 0
        assert key_padding_mask.shape == tokens.shape
        return key_padding_mask
    
    def _left_pad_by_padding_mask(self,
                                  tokens:torch.Tensor,
                                  key_padding_mask:Optional[torch.Tensor],
                                  pad_id:int
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        这个函数是用来根据填充掩码对输入的tokens进行左填充的。
        参数：
        tokens: 输入的token张量，形状为（batch_size, seq_len）。
        key_padding_mask: 填充掩码，形状为（batch_size, seq_len）。其中True表示有效的token，False表示需要填充的位置。
        pad_id: 用于填充的token ID。
        输出：
        new_tokens: 左填充后的token张量，形状为（batch_size, max_len）。其中max_len是根据key_padding_mask计算得到的最大有效长度。
        new_padding_mask: 左填充后的填充掩码，形状为（batch_size, max_len）。其中True表示有效的token，False表示需要填充的位置。
        """
        
        if key_padding_mask is None or key_padding_mask.all():
            return tokens, key_padding_mask

        lengths = key_padding_mask.sum(dim=1)
        max_len = int(lengths.max().item())

        # 向量化的好处在于不用每行做一次布尔索引，每次 lengths[i].item() 都要 GPU↔CPU 同步一下。batch 大或者 GPU 上跑就慢。
        # 用稳定排序把 padding(False) 排到左边、有效 token(True) 排到右边，相对顺序保持不变
        # sort_key: padding=0, valid=1；升序后 padding 在前，valid 在后
        sort_key = key_padding_mask.to(torch.int8)
        order = torch.argsort(sort_key, dim=1, stable=True)
        sorted_tokens = torch.gather(tokens, 1, order)
        sorted_mask   = torch.gather(key_padding_mask, 1, order)

        new_tokens = sorted_tokens[:, -max_len:].contiguous()
        new_padding_mask = sorted_mask[:, -max_len:].contiguous()

        new_tokens = torch.where(
            new_padding_mask,
            new_tokens,
            torch.full_like(new_tokens, pad_id),
        )
        return new_tokens, new_padding_mask
    
    
    def forward(self,
                input_ids:torch.Tensor,
                labels:Optional[torch.Tensor] = None,
                key_padding_mask:Optional[torch.Tensor] = None,
    ) -> CausalLMOutputWithPast:     
        """
        这个函数是用来执行Transformer模型的前向传播的。
        参数：
        input_ids: 输入的token ID张量，形状为（batch_size, seq_len）。
        labels: 可选的标签张量，形状为（batch_size, seq_len）。如果提供了labels，则会计算交叉熵损失。
        key_padding_mask: 可选的填充掩码，形状为（batch_size, seq_len）。其中True表示有效的token，False表示需要填充的位置。
        输出：  
        CausalLMOutputWithPast: 包含损失和logits的输出对象。损失是一个标量张量，表示交叉熵损失；logits是一个张量，形状为（batch_size, seq_len, vocab_size），表示每个位置的预测分布。
        """
        
        _, seq_len = input_ids.shape           
        x = self.token_embeddings(input_ids)
        x = self.dropout(x)
        
        for layer in self.layers:
            x = layer(
                x, 
                self.freqs_cos[:seq_len, :],
                self.freqs_sin[:seq_len, :], 
                key_padding_mask
            )
    
        x = self.RMSNorm(x)

        if labels is not None:
            logits = self.output(x)
            # 固定使用 -100 作为忽略标记。由 dataset 决定哪些位置不计入 loss
            # （预训练全部参与；SFT 阶段把 prompt 部分填 -100）
            main_loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                labels.view(-1),
                ignore_index=-100,
                reduction="mean"
            )
            # 汇总所有 MoE 层的负载均衡损失;dense 模型或非训练时为 0 tensor
            aux_loss = sum(
                (layer.FFN.loss for layer in self.layers
                 if isinstance(layer.FFN, MoEFeedForward)),
                start=x.new_tensor(0.0),
            )
            loss = main_loss + aux_loss
        else:
            logits = self.output(x[:, [-1], :])#采用left_padding
            loss = None
        
        return CausalLMOutputWithPast(loss, logits)    
        
    @torch.inference_mode()
    def generate(self,
                 input_ids:torch.Tensor,
                 max_new_tokens:int=256,
                 top_k:int=None,
                 temperature:float=1.0,
                 key_padding_mask:Optional[torch.Tensor]=None,
    ):
        """
        这个函数是用来生成文本的。
        参数：
        input_ids: 输入的token ID张量，形状为（batch_size, seq_len）。
        max_new_tokens: 要生成的新token的最大数量。
        top_k: 用于top-k采样，表示保留的最高概率token的数量。
        temperature: 用于调整生成文本的多样性，值越小越确定，值越大越随机。
        key_padding_mask: 可选的填充掩码，形状为（batch_size, seq_len）。其中True表示有效的token，False表示需要填充的位置。
        输出：
        output: 生成的token ID张量，形状为（batch_size, generated_seq_len）。其中generated_seq_len是根据max_new_tokens和输入序列长度计算得到的实际生成长度。
        """
        bs, _ = input_ids.shape
        
        if self.eos_token_id is not None:
            eos_id = self.eos_token_id
        else:
            raise ValueError(
                "eos_token_id is required for generation but was not set in the model config."
            )
        
        if self.pad_token_id is not None:
            pad_id = self.pad_token_id
        else:
            pad_id = self.eos_token_id
        
        key_padding_mask = self._prepare_padding_mask(key_padding_mask, input_ids)
        input_ids, key_padding_mask = \
            self._left_pad_by_padding_mask(input_ids, key_padding_mask, pad_id)
        
        finished = torch.zeros(bs, 1, dtype=torch.bool, device=input_ids.device)
        output = input_ids.new_empty((bs, 0))
        
        for _ in range(max_new_tokens):                    
            if input_ids.shape[1] > self.max_seq_len:
                input_ids = input_ids[:, -self.max_seq_len:]
                if key_padding_mask is not None:
                    key_padding_mask = key_padding_mask[:, -self.max_seq_len:]
            
            logits = self(input_ids, key_padding_mask=key_padding_mask).logits[:, -1, :]
            
            if temperature == 0.0:
                _, next_tokens = torch.topk(logits, k=1, dim=-1)
            else:
                logits = logits / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, k=min(top_k, logits.size(-1)), dim=-1)
                    logits[logits < v[:, [-1]]] = float('-inf')
                probs = torch.softmax(logits, dim=-1)
                next_tokens = torch.multinomial(probs, 1)
        
            if key_padding_mask is not None:
                next_mask = torch.ones((bs, 1), dtype=torch.bool, device=input_ids.device)
                if finished.any():
                    next_mask[finished==True] = False
                key_padding_mask = torch.cat([key_padding_mask, next_mask], dim=-1)             
        
            fill_token = pad_id
            fill = torch.full_like(next_tokens, fill_token)
            next_tokens = torch.where(finished, fill, next_tokens)
            finished = finished | (next_tokens == eos_id)

            output = torch.cat([output, next_tokens], dim=-1)
            input_ids = torch.cat([input_ids, next_tokens], dim=-1)
            
            if finished.all():
                break
        
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
    
    """这个代码块是用来测试TransformerBlock类的功能的。"""
    # args = MyModelConfig()
    
    # freqs_cos, freqs_sin = precompute_freqs_cs(args.max_seq_len, args.dim // args.n_heads)
    
    # transformer_block_model = TransformerBlock(0, args)
    
    # bs = 1
    # seq_len = 50
    # X = torch.randn(bs, seq_len, args.dim)
    # output = transformer_block_model(X, freqs_cos[:seq_len], freqs_sin[:seq_len])
    # print("output shape", output.shape)
    
    """这个代码块是用来测试Transformer类的功能的。"""
    # args = MyModelConfig()
    # transformer_model = Transformer(args)
    # bs = 1
    # seq_len = 50
    # input_ids = torch.randint(0, args.vocab_size, (bs, seq_len))
    # output = transformer_model(input_ids)
    # print("output shape", output.logits.shape)