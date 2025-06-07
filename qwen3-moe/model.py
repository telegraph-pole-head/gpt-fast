# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F


def find_multiple(n: int, k: int) -> int:
    if n % k == 0:
        return n
    return n + k - (n % k)


@dataclass
class ModelArgs:
    block_size: int = 40960
    vocab_size: int = 151936
    n_layer: int = 48
    n_head: int = 32
    dim: int = 2048
    intermediate_size: int = 6144
    n_local_heads: int = 4
    head_dim: int = 128
    rope_base: float = 1000000.0
    norm_eps: float = 1e-6
    num_experts: int = 128
    num_activated_experts: int = 8
    moe_intermediate_size: int = 768

    def __post_init__(self):
        if self.intermediate_size is None:
            hidden_dim = 4 * self.dim
            n_hidden = int(2 * hidden_dim / 3)
            self.intermediate_size = find_multiple(n_hidden, 256)

    @classmethod
    def from_name(cls, name: str):
        if name in transformer_configs:
            return cls(**transformer_configs[name])
        # fuzzy search
        config = [
            config
            for config in transformer_configs
            if config in str(name).upper() or config in str(name)
        ]
        assert len(config) == 1, name
        return cls(**transformer_configs[config[0]])


transformer_configs = {
    "Qwen3-30B-A3B": dict(
        block_size=40960,
        n_layer=48,
        n_head=32,
        n_local_heads=4,
        dim=2048,
        intermediate_size=6144,
        head_dim=128,
        rope_base=1000000.0,
        num_experts=128,
        num_activated_experts=8,
        moe_intermediate_size=768,
        vocab_size=151936,
        norm_eps=1e-6,
    ),
}


class KVCache(nn.Module):
    def __init__(
        self, max_batch_size, max_seq_length, n_heads, head_dim, dtype=torch.bfloat16
    ):
        super().__init__()
        cache_shape = (max_batch_size, n_heads, max_seq_length, head_dim)
        self.register_buffer("k_cache", torch.zeros(cache_shape, dtype=dtype))
        self.register_buffer("v_cache", torch.zeros(cache_shape, dtype=dtype))

    def update(self, input_pos, k_val, v_val):
        # input_pos: [S], k_val: [B, H, S, D]
        assert input_pos.shape[0] == k_val.shape[2]

        k_out = self.k_cache
        v_out = self.v_cache
        k_out[:, :, input_pos] = k_val
        v_out[:, :, input_pos] = v_val

        return k_out, v_out


class Transformer(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.config = config

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList(
            TransformerBlock(config) for _ in range(config.n_layer)
        )
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)

        self.freqs_cis: Optional[Tensor] = None
        self.mask_cache: Optional[Tensor] = None
        self.max_batch_size = -1
        self.max_seq_length = -1

    def setup_caches(self, max_batch_size, max_seq_length):
        if (
            self.max_seq_length >= max_seq_length
            and self.max_batch_size >= max_batch_size
        ):
            return
        max_seq_length = find_multiple(max_seq_length, 8)
        self.max_seq_length = max_seq_length
        self.max_batch_size = max_batch_size
        for b in self.layers:
            b.attention.kv_cache = KVCache(
                max_batch_size,
                max_seq_length,
                self.config.n_local_heads,
                self.config.head_dim,
            )

        self.freqs_cis = precompute_freqs_cis(
            self.config.block_size, self.config.head_dim, self.config.rope_base
        )
        self.causal_mask = torch.tril(
            torch.ones(self.max_seq_length, self.max_seq_length, dtype=torch.bool)
        )

    def forward(self, idx: Tensor, input_pos: Optional[Tensor] = None) -> Tensor:
        assert self.freqs_cis is not None, "Caches must be initialized first"
        mask = self.causal_mask[None, None, input_pos]
        freqs_cis = self.freqs_cis[input_pos]
        x = self.tok_embeddings(idx)

        for i, layer in enumerate(self.layers):
            x = layer(x, input_pos, freqs_cis, mask)
        x = self.norm(x)
        logits = self.output(x)
        return logits

    @classmethod
    def from_name(cls, name: str):
        return cls(ModelArgs.from_name(name))


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.attention = Attention(config)
        self.block_sparse_moe = MOEFeedForward(config)
        self.ffn_norm = RMSNorm(config.dim, config.norm_eps)
        self.attention_norm = RMSNorm(config.dim, config.norm_eps)

    def forward(
        self, x: Tensor, input_pos: Tensor, freqs_cis: Tensor, mask: Tensor
    ) -> Tensor:
        h = x + self.attention(self.attention_norm(x), freqs_cis, mask, input_pos)
        out = h + self.block_sparse_moe(self.ffn_norm(h))
        return out


class Attention(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        assert config.dim % config.n_head == 0

        # Separate projections for Q, K, V
        self.q_proj = nn.Linear(config.dim, config.n_head * config.head_dim, bias=False)
        self.k_proj = nn.Linear(
            config.dim, config.n_local_heads * config.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            config.dim, config.n_local_heads * config.head_dim, bias=False
        )
        self.o_proj = nn.Linear(config.n_head * config.head_dim, config.dim, bias=False)

        # RMS norm for Q and K (Qwen3 feature)
        self.q_norm = RMSNorm(config.head_dim, eps=config.norm_eps)
        self.k_norm = RMSNorm(config.head_dim, eps=config.norm_eps)

        self.kv_cache = None

        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.n_local_heads = config.n_local_heads
        self.dim = config.dim
        self.scaling = config.head_dim**-0.5

    def forward(
        self,
        x: Tensor,
        freqs_cis: Tensor,
        mask: Tensor,
        input_pos: Optional[Tensor] = None,
    ) -> Tensor:
        bsz, seqlen, _ = x.shape

        # Project Q, K, V separately
        q = self.q_proj(x).view(bsz, seqlen, self.n_head, self.head_dim)
        k = self.k_proj(x).view(bsz, seqlen, self.n_local_heads, self.head_dim)
        v = self.v_proj(x).view(bsz, seqlen, self.n_local_heads, self.head_dim)

        # Apply RMS norm to Q and K
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Apply rotary embeddings
        q = apply_rotary_emb(q, freqs_cis)
        k = apply_rotary_emb(k, freqs_cis)

        q, k, v = map(lambda x: x.transpose(1, 2), (q, k, v))

        if self.kv_cache is not None:
            k, v = self.kv_cache.update(input_pos, k, v)

        # Repeat K and V for grouped query attention
        k = k.repeat_interleave(self.n_head // self.n_local_heads, dim=1)
        v = v.repeat_interleave(self.n_head // self.n_local_heads, dim=1)

        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)

        y = (
            y.transpose(1, 2)
            .contiguous()
            .view(bsz, seqlen, self.n_head * self.head_dim)
        )

        y = self.o_proj(y)
        return y


class ConditionalFeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Use moe_intermediate_size for MoE experts
        self.w1 = nn.Parameter(
            torch.empty(config.num_experts, config.moe_intermediate_size, config.dim)
        )
        self.w2 = nn.Parameter(
            torch.empty(config.num_experts, config.dim, config.moe_intermediate_size)
        )
        self.w3 = nn.Parameter(
            torch.empty(config.num_experts, config.moe_intermediate_size, config.dim)
        )

    def forward(self, x: Tensor, expert_indices: Tensor) -> Tensor:
        w1_weights = self.w1[expert_indices]  # [T, A, D, D]
        w3_weights = self.w3[expert_indices]  # [T, A, D, D]
        w2_weights = self.w2[expert_indices]  # [T, A, D, D]
        x1 = F.silu(torch.einsum("ti,taoi -> tao", x, w1_weights))
        x3 = torch.einsum("ti, taoi -> tao", x, w3_weights)
        expert_outs = torch.einsum("tao, taio -> tai", (x1 * x3), w2_weights)
        return expert_outs


class MOEFeedForward(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.gate = nn.Linear(config.dim, config.num_experts, bias=False)
        self.cond_ffn = ConditionalFeedForward(config)
        self.dim = config.dim
        self.num_activated_experts = config.num_activated_experts

    def forward(self, x: Tensor) -> Tensor:
        original_shape = x.shape
        x = x.view(-1, self.dim)
        # T = num_tokens, E = num_experts, D = hidden dim, A = activated experts
        # x: [T, D]
        scores = self.gate(x)  # [T, E]
        expert_weights = F.softmax(scores, dim=-1, dtype=torch.float32)
        expert_weights, expert_indices = torch.topk(
            expert_weights, self.num_activated_experts, dim=-1
        )  # [T, A], [T, A]
        # Qwen3 normalizes topk probabilities
        expert_weights /= expert_weights.sum(dim=-1, keepdim=True)  # [T, A]
        expert_weights = expert_weights.to(x.dtype)
        expert_outs = self.cond_ffn(x, expert_indices)
        result = torch.einsum("tai,ta -> ti", expert_outs, expert_weights)
        return result.view(original_shape)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)

    def forward(self, x: Tensor) -> Tensor:
        input_dtype = x.dtype
        x = x.to(torch.float32)
        output = self._norm(x)
        return (output * self.weight).to(input_dtype)


def precompute_freqs_cis(seq_len: int, n_elem: int, base: int = 1000000) -> Tensor:
    freqs = 1.0 / (
        base ** (torch.arange(0, n_elem, 2)[: (n_elem // 2)].float() / n_elem)
    )
    t = torch.arange(seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    cache = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1)
    return cache.to(dtype=torch.bfloat16)


def apply_rotary_emb(x: Tensor, freqs_cis: Tensor) -> Tensor:
    xshaped = x.float().reshape(*x.shape[:-1], -1, 2)
    freqs_cis = freqs_cis.view(1, xshaped.size(1), 1, xshaped.size(3), 2)
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1],
            xshaped[..., 1] * freqs_cis[..., 0] + xshaped[..., 0] * freqs_cis[..., 1],
        ],
        -1,
    )

    x_out2 = x_out2.flatten(3)
    return x_out2.type_as(x)
