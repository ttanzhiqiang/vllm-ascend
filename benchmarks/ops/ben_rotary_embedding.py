# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
from typing import Optional, Tuple, Union

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch_npu
import vllm
import vllm_ascend.platform


def benchmark_npu(fn, num_iterations=100, num_warmup_iterations=50):
    """
    Benchmark function for NPU operations
    
    Args:
        fn: Function to benchmark
        num_iterations: Number of timing iterations
        num_warmup_iterations: Number of warmup iterations
    
    Returns:
        float: Minimum elapsed time in seconds
    """
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    times = np.zeros(num_iterations + num_warmup_iterations)

    # Run iterations
    for i in range(num_warmup_iterations + num_iterations):
        with torch.no_grad():
            start.record()
            fn()  # Execute the function
            end.record()
        torch.npu.synchronize()
        times[i] = start.elapsed_time(end)

    # Remove warmup iterations and convert to seconds
    times = times[num_warmup_iterations:]
    elapsed_time = np.amin(times) / 1000
    return elapsed_time


def _apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    is_neox_style: bool,
) -> torch.Tensor:
    """
    Args:
        x: [num_tokens, num_heads, head_size]
        cos: [num_tokens, head_size // 2]
        sin: [num_tokens, head_size // 2]
        is_neox_style: Whether to use the Neox-style or GPT-J-style rotary
            positional embeddings.
    """
    cos = cos.unsqueeze(-2).to(x.dtype)
    sin = sin.unsqueeze(-2).to(x.dtype)
    if is_neox_style:
        x1, x2 = torch.chunk(x, 2, dim=-1)
    else:
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    if is_neox_style:
        return torch.cat((o1, o2), dim=-1)
    else:
        return torch.stack((o1, o2), dim=-1).flatten(-2)


class RotaryEmbedding(nn.Module):
    """Original rotary positional embedding."""

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: int,
        is_neox_style: bool,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        self.rotary_dim = rotary_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.is_neox_style = is_neox_style
        self.dtype = dtype

        cache = self._compute_cos_sin_cache()
        cache = cache.to(dtype)
        self.cos_sin_cache: torch.Tensor
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    def _compute_inv_freq(self, base: Union[int, float]) -> torch.Tensor:
        """Compute the inverse frequency."""
        inv_freq = 1.0 / (base**(torch.arange(
            0, self.rotary_dim, 2, dtype=torch.float) / self.rotary_dim))
        return inv_freq

    def _compute_cos_sin_cache(self) -> torch.Tensor:
        """Compute the cos and sin cache."""
        inv_freq = self._compute_inv_freq(self.base)
        t = torch.arange(self.max_position_embeddings, dtype=torch.float)

        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1)
        return cache

    def forward_native(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        offsets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """A PyTorch-native implementation of forward()."""
        if offsets is not None:
            positions = positions + offsets
        positions = positions.flatten()
        num_tokens = positions.shape[0]
        cos_sin = self.cos_sin_cache.index_select(0, positions)
        cos, sin = cos_sin.chunk(2, dim=-1)

        query_shape = query.shape
        query = query.view(num_tokens, -1, self.head_size)
        query_rot = query[..., :self.rotary_dim]
        query_pass = query[..., self.rotary_dim:]
        query_rot = _apply_rotary_emb(query_rot, cos, sin, self.is_neox_style)
        query = torch.cat((query_rot, query_pass), dim=-1).reshape(query_shape)

        key_shape = key.shape
        key = key.view(num_tokens, -1, self.head_size)
        key_rot = key[..., :self.rotary_dim]
        key_pass = key[..., self.rotary_dim:]
        key_rot = _apply_rotary_emb(key_rot, cos, sin, self.is_neox_style)
        key = torch.cat((key_rot, key_pass), dim=-1).reshape(key_shape)
        return query, key

IS_NEOX_STYLE = [True]
DTYPES = [torch.half]
HEAD_SIZE = 64
NUM_QUERY_HEADS = [32]
NUM_KEY_HEADS = [1]
SEQ_CONFIG = [
    (1, 4103),
    (4, 4103),
    (16, 4103),
    (64, 4103),
    (256, 4103),
    (512, 4103),
    (1024, 4103),     # (seq_len, cache_size)
    (4091, 4096),
    (8192, 4116),
]
SEEDS = [0]
DEVICES = [f"npu:{0}"]
DEFAULT_ATOL = 1e-3
DEFAULT_RTOL = 1e-3

@pytest.mark.parametrize("is_neox_style", IS_NEOX_STYLE)
@pytest.mark.parametrize("seq_len,cache_size", SEQ_CONFIG)
@pytest.mark.parametrize("num_query_heads", NUM_QUERY_HEADS)
@pytest.mark.parametrize("num_key_heads", NUM_KEY_HEADS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", DEVICES)
@torch.inference_mode()
def test_rotary_embedding(
    is_neox_style: bool,
    seq_len: int,
    cache_size: int,
    num_query_heads: int,
    num_key_heads: int,
    dtype: torch.dtype,
    seed: int,
    device: str,
    base: int = 10000,
    head_size: int = HEAD_SIZE,
) -> None:
    # Set random seed and device
    torch.manual_seed(seed)
    torch.set_default_device(device)

    # Initialize rotary embedding with correct cache size
    rope = RotaryEmbedding(
        head_size=head_size,
        rotary_dim=head_size,
        max_position_embeddings=cache_size,
        base=base,
        is_neox_style=is_neox_style,
        dtype=dtype
    )
    rope = rope.to(dtype=dtype)
    
    # Generate input tensors with correct shapes
    positions = torch.randint(0, cache_size, (seq_len,))
    query = torch.randn(seq_len, num_query_heads, head_size, dtype=dtype)
    key = torch.randn(seq_len, num_key_heads, head_size, dtype=dtype)

    # Define reference function
    def ref_fn():
        return rope.forward_native(positions, query, key)

    # Define custom function 
    def custom_fn():
        return torch.ops._C.rotary_embedding(
            positions,
            query, 
            key,
            rope.head_size,
            rope.cos_sin_cache,
            rope.is_neox_style,
        )

    # Get results for correctness testing
    ref_query, ref_key = ref_fn()
    custom_query, custom_key = custom_fn()

    # Benchmark both implementations
    ref_time = benchmark_npu(ref_fn)
    custom_time = benchmark_npu(custom_fn)

    # Print performance results
    print("\nTest Configuration:")
    print(f"Sequence length: {seq_len}")
    print(f"Cache size: {cache_size}")
    print(f"Query heads: {num_query_heads}")
    print(f"Key heads: {num_key_heads}")
    print(f"Head size: {head_size}")
    
    print("\nTensor shapes:")
    print(f"Positions: {positions.shape}")
    print(f"Query: {query.shape}")
    print(f"Key: {key.shape}")
    print(f"Cos/Sin cache: {rope.cos_sin_cache.shape}")

    print("\nPerformance Results:")
    print(f"Reference implementation: {ref_time*1000:.3f} ms")
    print(f"Custom implementation: {custom_time*1000:.3f} ms") 
    print(f"Speedup: {ref_time/custom_time:.2f}x")

    # Compare results for correctness
    torch.testing.assert_close(
        custom_query,
        ref_query,
        atol=DEFAULT_ATOL,
        rtol=DEFAULT_RTOL,
        msg="Query results mismatch"
    )
    torch.testing.assert_close(
        custom_key,
        ref_key, 
        atol=DEFAULT_ATOL,
        rtol=DEFAULT_RTOL,
        msg="Key results mismatch"
    )

if __name__ == "__main__":
    pytest.main([__file__])
