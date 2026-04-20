"""Spike-driven self-attention for event camera detection.

Based on Spikformer (Zhou et al., ICLR 2023) spike-driven attention,
adapted for streaming event camera processing with RVT-style window/grid
partition for multi-scale spatial mixing.

Key properties:
    - Q, K, V are binary spikes (pass through stateful LIF after projection).
    - attn = Q @ K^T: integer coincidence count (no softmax).
    - out = attn @ V: sum of coincidence-weighted binary V vectors.
    - No softmax, no exp, no division — addition + integer multiply only.

Stateful LIF (critical for streaming):
    The LIF applied after qkv projection must maintain membrane state across
    timesteps. Otherwise, at streaming time with small per-step inputs, the
    projection output never reaches threshold → Q, K, V are all zeros → no
    gradient flows through attention.

State management:
    LIF state tensors are stored in UNPARTITIONED shape (N, H, W, C) so that
    RVT's `recursive_reset` (which applies batch indices to dim 0) works.
    Inside the attention forward, state is partitioned alongside input,
    then reverse-partitioned before return.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from .maxvit.maxvit import (
    window_partition, window_reverse,
    grid_partition, grid_reverse,
    PartitionType,
)
from .spiking import LIFNeuron


# Attention LIF state: (q_mem, k_mem, v_mem) each (N, H, W, C) or None.
AttnState = Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]


class SpikeSelfAttention(nn.Module):
    """Multi-head spike-driven self-attention with stateful LIF on Q, K, V.

    All inputs and mems are partitioned (channel-last, (B', P_h, P_w, C)).
    Membrane state flows in and out of this module.
    """

    def __init__(self,
                 dim: int,
                 dim_head: int = 32,
                 qkv_bias: bool = False,
                 beta_init: float = 0.9,
                 learn_beta: bool = True,
                 threshold: float = 1.0,
                 surrogate: str = 'triangle'):
        super().__init__()
        assert dim % dim_head == 0, \
            f"dim ({dim}) must be divisible by dim_head ({dim_head})"
        self.num_heads = dim // dim_head
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        # Stateful LIF per Q/K/V stream (scalar beta, shape-agnostic).
        lif_kwargs = dict(
            beta_init=beta_init,
            learn_beta=learn_beta,
            threshold=threshold,
            reset_mechanism='subtract',
            channels=None,
            surrogate=surrogate,
        )
        self.lif_q = LIFNeuron(**lif_kwargs)
        self.lif_k = LIFNeuron(**lif_kwargs)
        self.lif_v = LIFNeuron(**lif_kwargs)

        self.proj = nn.Linear(dim, dim, bias=qkv_bias)

    def forward(self,
                x: torch.Tensor,
                q_mem: Optional[torch.Tensor] = None,
                k_mem: Optional[torch.Tensor] = None,
                v_mem: Optional[torch.Tensor] = None,
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x:     (B', P_h, P_w, C) partitioned input
            q/k/v_mem: (B', P_h, P_w, C) partitioned membrane state or None

        Returns:
            out:    (B', P_h, P_w, C) coincidence-weighted feature (float)
            new_q_mem, new_k_mem, new_v_mem: updated membrane states, partitioned
        """
        B = x.shape[0]
        restore_shape = x.shape[:-1]           # (B', P_h, P_w)
        C = x.shape[-1]

        # Linear projection: (B', P_h, P_w, 3C)
        qkv = self.qkv(x)
        # Flatten spatial for attention: (B', L, 3C) where L = P_h * P_w
        qkv_flat = qkv.view(B, -1, 3 * C)
        q_float, k_float, v_float = qkv_flat.chunk(3, dim=-1)   # each (B', L, C)

        # Reshape mems to match (B', L, C) for LIF
        if q_mem is not None:
            q_mem = q_mem.view(B, -1, C)
        if k_mem is not None:
            k_mem = k_mem.view(B, -1, C)
        if v_mem is not None:
            v_mem = v_mem.view(B, -1, C)

        # Stateful LIF: float → binary spike + membrane update
        q_spike, q_mem_new = self.lif_q(q_float, q_mem)
        k_spike, k_mem_new = self.lif_k(k_float, k_mem)
        v_spike, v_mem_new = self.lif_v(v_float, v_mem)

        # Multi-head reshape: (B', L, C) → (B', heads, L, dim_head)
        def to_heads(t):
            return t.view(B, -1, self.num_heads, self.dim_head).transpose(1, 2)
        q = to_heads(q_spike)
        k = to_heads(k_spike)
        v = to_heads(v_spike)

        # Spike-driven attention: Q, K binary → Q @ K^T is integer coincidence.
        attn = (q @ k.transpose(-2, -1)) * self.scale     # (B', heads, L, L)
        out = attn @ v                                      # (B', heads, L, dim_head)

        # Merge heads: (B', heads, L, dim_head) → (B', L, C)
        out = out.transpose(1, 2).reshape(B, -1, C)
        out = self.proj(out)

        # Restore partitioned spatial shape for output and mems
        out = out.view(*restore_shape, C)
        q_mem_new = q_mem_new.view(*restore_shape, C)
        k_mem_new = k_mem_new.view(*restore_shape, C)
        v_mem_new = v_mem_new.view(*restore_shape, C)

        return out, q_mem_new, k_mem_new, v_mem_new


class SpikePartitionAttention(nn.Module):
    """RVT-style partition (window or grid) + SpikeSelfAttention.

    Input/output: (N, H, W, C) channel-last float.
    State: (q_mem, k_mem, v_mem) each (N, H, W, C), UNPARTITIONED (for RVT
    recursive_reset batch-index compatibility).
    """

    def __init__(self,
                 dim: int,
                 partition_type: PartitionType,
                 partition_size: Tuple[int, int],
                 dim_head: int = 32,
                 qkv_bias: bool = False,
                 beta_init: float = 0.9,
                 learn_beta: bool = True,
                 threshold: float = 1.0,
                 surrogate: str = 'triangle'):
        super().__init__()
        self.partition_window = (partition_type == PartitionType.WINDOW)
        if isinstance(partition_size, int):
            partition_size = (partition_size, partition_size)
        else:
            partition_size = tuple(partition_size)
            assert len(partition_size) == 2
        self.partition_size = partition_size

        self.attn = SpikeSelfAttention(
            dim=dim,
            dim_head=dim_head,
            qkv_bias=qkv_bias,
            beta_init=beta_init,
            learn_beta=learn_beta,
            threshold=threshold,
            surrogate=surrogate,
        )

    def _partition(self, t: torch.Tensor) -> torch.Tensor:
        if self.partition_window:
            return window_partition(t, self.partition_size)
        return grid_partition(t, self.partition_size)

    def _reverse(self, t: torch.Tensor, img_size: Tuple[int, int]) -> torch.Tensor:
        if self.partition_window:
            return window_reverse(t, self.partition_size, img_size)
        return grid_reverse(t, self.partition_size, img_size)

    def forward(self,
                x: torch.Tensor,
                state: AttnState = None,
                ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: (N, H, W, C)
            state: (q_mem, k_mem, v_mem) each (N, H, W, C), unpartitioned.

        Returns:
            out: (N, H, W, C)
            new_state: (q_mem, k_mem, v_mem) each (N, H, W, C)
        """
        img_size = (x.shape[1], x.shape[2])

        # Init state with zeros matching x shape if absent.
        if state is None:
            q_mem = torch.zeros_like(x)
            k_mem = torch.zeros_like(x)
            v_mem = torch.zeros_like(x)
        else:
            q_mem, k_mem, v_mem = state

        # Partition input AND state tensors alongside.
        x_part = self._partition(x)
        q_mem_part = self._partition(q_mem)
        k_mem_part = self._partition(k_mem)
        v_mem_part = self._partition(v_mem)

        # Run spike attention on partitioned tensors.
        out_part, q_mem_new_part, k_mem_new_part, v_mem_new_part = self.attn(
            x_part, q_mem_part, k_mem_part, v_mem_part)

        # Reverse partition for output AND new mems.
        out = self._reverse(out_part, img_size)
        q_mem_new = self._reverse(q_mem_new_part, img_size)
        k_mem_new = self._reverse(k_mem_new_part, img_size)
        v_mem_new = self._reverse(v_mem_new_part, img_size)

        return out, (q_mem_new, k_mem_new, v_mem_new)
