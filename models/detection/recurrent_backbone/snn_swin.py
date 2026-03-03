"""SNN Swin Backbone: Spiking Swin Transformer with CGRA Window Attention.

Hierarchical 4-stage backbone producing multi-scale feature maps for detection.
Uses CGRA coincidence-gated attention (QK^T gate with attention cell LIF) and
Swin's shifted-window scheme. All LIF membranes persist across timesteps via
explicit state passing compatible with RVT's RNNStates.

Reference: /home/kwdahun/PRISM/src/networks/spiking_swin.py
           /home/kwdahun/PRISM/src/layers/cgra_window_attention.py
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from omegaconf import DictConfig
from timm.layers import trunc_normal_

from data.utils.types import BackboneFeatures, FeatureMap
from models.layers.spiking import LIFNeuron
from modules.utils.detection import RNNStates
from .base import BaseDetector

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
SwinSubState = List[torch.Tensor]   # flat list of membrane tensors per module
SwinStates = List[SwinSubState]     # [embed_state, stage0, stage1, stage2, stage3]

# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """Partition (B, H, W, C) into (B*nW, Ws, Ws, C) windows."""
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)


def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int) -> torch.Tensor:
    """Reverse window_partition: (B*nW, Ws, Ws, C) -> (B, H, W, C)."""
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)


def compute_sw_mask(
    H: int, W: int, window_size: int, shift_size: int, device: torch.device | None = None,
) -> torch.Tensor:
    """Shifted-window attention mask. Returns (nW, N, N) with -100 for masked positions."""
    img_mask = torch.zeros((1, H, W, 1), device=device)
    h_slices = (slice(0, -window_size), slice(-window_size, -shift_size), slice(-shift_size, None))
    w_slices = (slice(0, -window_size), slice(-window_size, -shift_size), slice(-shift_size, None))
    cnt = 0
    for h in h_slices:
        for w in w_slices:
            img_mask[:, h, w, :] = cnt
            cnt += 1
    mask_windows = window_partition(img_mask, window_size)           # (nW, Ws, Ws, 1)
    mask_windows = mask_windows.view(-1, window_size * window_size)  # (nW, N)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)  # (nW, N, N)
    return attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(attn_mask == 0, 0.0)


# ---------------------------------------------------------------------------
# Sub-modules
# ---------------------------------------------------------------------------

class BNForTokens(nn.Module):
    """BatchNorm1d for (B*nW, N, C) token sequences."""

    def __init__(self, num_features: int):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(x.transpose(-1, -2)).transpose(-1, -2)


class CGRAWindowAttention(nn.Module):
    """CGRA coincidence-gated window attention with explicit state passing.

    6 LIF neurons per module. All membranes are stored batch-first and reshaped
    during forward for correct window-parallel computation.

    State list (6 tensors):
        [0] q_mem:    (B, nW*N, C)
        [1] k_mem:    (B, nW*N, C)
        [2] v_mem:    (B, nW*N, C)
        [3] cell_mem: (B, nW*H, N, N)   — attention cell LIF (hard reset)
        [4] attn_mem: (B, nW*N, C)
        [5] out_mem:  (B, nW*N, C)
    """

    NUM_STATES = 6

    def __init__(
        self,
        dim: int,
        window_size: Tuple[int, int],
        num_heads: int,
        attn_scale: float = 0.125,
        output_scale: float = 0.25,
        use_rel_pos_bias: bool = True,
        snn_cfg: Optional[DictConfig] = None,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.window_size = window_size   # (Wh, Ww)
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.attn_scale = attn_scale
        self.output_scale = output_scale
        self.use_rel_pos_bias = use_rel_pos_bias
        self.N = window_size[0] * window_size[1]  # tokens per window

        beta_init = snn_cfg.get('beta_init', 0.5) if snn_cfg else 0.5
        learn_beta = snn_cfg.get('learn_beta', True) if snn_cfg else True
        threshold = snn_cfg.get('threshold', 1.0) if snn_cfg else 1.0

        # Q/K/V projections
        self.proj_q = nn.Linear(dim, dim)
        self.bn_q = BNForTokens(dim)
        self.lif_q = LIFNeuron(beta_init=beta_init, learn_beta=learn_beta, threshold=threshold)

        self.proj_k = nn.Linear(dim, dim)
        self.bn_k = BNForTokens(dim)
        self.lif_k = LIFNeuron(beta_init=beta_init, learn_beta=learn_beta, threshold=threshold)

        self.proj_v = nn.Linear(dim, dim)
        self.bn_v = BNForTokens(dim)
        self.lif_v = LIFNeuron(beta_init=beta_init, learn_beta=learn_beta, threshold=threshold)

        # Attention cell LIF: hard reset (zero), no input decay
        self.attn_cell_lif = LIFNeuron(
            beta_init=beta_init, learn_beta=learn_beta,
            threshold=threshold, reset_mechanism='zero',
        )

        # Cell state feedback projection
        self.proj_cs_feedback = nn.Linear(num_heads, dim)

        # Attention output LIF (lower threshold)
        self.attn_lif = LIFNeuron(
            beta_init=beta_init, learn_beta=learn_beta, threshold=0.5,
        )

        # Output projection
        self.proj_out = nn.Linear(dim, dim)
        self.bn_out = BNForTokens(dim)
        self.lif_out = LIFNeuron(beta_init=beta_init, learn_beta=learn_beta, threshold=threshold)

        # Relative position bias table (optional)
        if use_rel_pos_bias:
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
            )
            trunc_normal_(self.relative_position_bias_table, std=0.02)

            coords_h = torch.arange(window_size[0])
            coords_w = torch.arange(window_size[1])
            coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij'))
            coords_flat = coords.flatten(1)
            relative_coords = coords_flat[:, :, None] - coords_flat[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += window_size[0] - 1
            relative_coords[:, :, 1] += window_size[1] - 1
            relative_coords[:, :, 0] *= 2 * window_size[1] - 1
            self.register_buffer('relative_position_index', relative_coords.sum(-1))

    def _get_rel_pos_bias(self) -> torch.Tensor:
        """Returns (num_heads, N, N) relative position bias."""
        N = self.N
        bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(N, N, -1)
        return bias.permute(2, 0, 1).contiguous()

    def forward(
        self,
        x: torch.Tensor,
        prev_mems: Optional[List[Optional[torch.Tensor]]],
        B: int,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            x: (B*nW, N, C) windowed tokens.
            prev_mems: list of 6 batch-first membrane tensors, or None.
            B: batch size (for reshape between storage and forward shapes).
            mask: (nW, N, N) shifted-window mask or None.
        Returns:
            output: (B*nW, N, C)
            new_mems: list of 6 batch-first membrane tensors.
        """
        BnW, N, C = x.shape
        H_heads, d = self.num_heads, self.head_dim
        nW = BnW // B

        # Unpack previous membranes (storage -> forward shape)
        if prev_mems is None:
            prev_mems = [None] * self.NUM_STATES
        p_q, p_k, p_v, p_cell, p_attn, p_out = prev_mems

        # Reshape batch-first -> window-parallel
        def to_fwd(mem: Optional[torch.Tensor], target_shape: Tuple[int, ...]) -> Optional[torch.Tensor]:
            if mem is None:
                return None
            return mem.reshape(target_shape)

        p_q = to_fwd(p_q, (BnW, N, C))
        p_k = to_fwd(p_k, (BnW, N, C))
        p_v = to_fwd(p_v, (BnW, N, C))
        p_cell = to_fwd(p_cell, (BnW * H_heads, N * N))
        p_attn = to_fwd(p_attn, (BnW, N, C))
        p_out = to_fwd(p_out, (BnW, N, C))

        # Cell state feedback from previous attention cell membrane
        if p_cell is not None:
            cs_mem = p_cell.reshape(BnW, H_heads, N, N)
            cs_summary = cs_mem.mean(dim=-1).permute(0, 2, 1).contiguous()  # (BnW, N, H)
            fb = self.proj_cs_feedback(cs_summary)  # (BnW, N, C)
        else:
            fb = 0

        # Q/K/V projections + feedback + LIF
        q_pre = self.bn_q(self.proj_q(x)) + fb
        q_spike, q_mem = self.lif_q(q_pre.contiguous(), p_q)
        q = q_spike.reshape(BnW, N, H_heads, d).permute(0, 2, 1, 3).contiguous()

        k_pre = self.bn_k(self.proj_k(x)) + fb
        k_spike, k_mem = self.lif_k(k_pre.contiguous(), p_k)
        k = k_spike.reshape(BnW, N, H_heads, d).permute(0, 2, 1, 3).contiguous()

        v_pre = self.bn_v(self.proj_v(x)) + fb
        v_spike, v_mem = self.lif_v(v_pre.contiguous(), p_v)
        v = v_spike.reshape(BnW, N, H_heads, d).permute(0, 2, 1, 3).contiguous()

        # Coincidence gate: Q @ K^T (always >= 0 since spikes are non-negative)
        gate = (q @ k.transpose(-2, -1)) * self.attn_scale  # (BnW, H, N, N)

        # Add relative position bias (optional)
        if self.use_rel_pos_bias:
            gate = gate + self._get_rel_pos_bias().unsqueeze(0)

        # Apply shifted-window mask (zero invalid pairs before LIF)
        if mask is not None:
            nW_mask = mask.shape[0]
            valid = (mask == 0).unsqueeze(1).unsqueeze(0)  # (1, nW, 1, N, N)
            gate = (gate.view(-1, nW_mask, H_heads, N, N) * valid).reshape(-1, H_heads, N, N)

        # Attention cell LIF (hard reset, fire-and-reset dynamics)
        gate_flat = gate.reshape(BnW * H_heads, N * N)
        cell_spike, cell_mem = self.attn_cell_lif(gate_flat, p_cell)
        attn_spike = cell_spike.reshape(BnW, H_heads, N, N)

        # Binary attention applied to V
        out = (attn_spike @ v) * self.output_scale
        out = out.transpose(1, 2).reshape(BnW, N, C).contiguous()

        # Attention output LIF
        out, attn_mem = self.attn_lif(out, p_attn)

        # Output projection
        out = self.bn_out(self.proj_out(out))
        out, out_mem = self.lif_out(out, p_out)

        # Reshape new membranes: forward -> batch-first storage
        def to_store(mem: torch.Tensor) -> torch.Tensor:
            return mem.reshape(B, -1, *mem.shape[1:]) if mem.dim() >= 2 else mem.reshape(B, -1)

        # For token-shaped mems (BnW, N, C) -> (B, nW*N, C)
        new_mems = [
            q_mem.reshape(B, nW * N, C),
            k_mem.reshape(B, nW * N, C),
            v_mem.reshape(B, nW * N, C),
            cell_mem.reshape(B, nW * H_heads, N, N),
            attn_mem.reshape(B, nW * N, C),
            out_mem.reshape(B, nW * N, C),
        ]

        return out, new_mems


class SpikingPatchEmbed(nn.Module):
    """Spiking Patch Embedding: 4x spatial downsample via 2 strided convs + RPE.

    State list (3 tensors): [lif1_mem, lif2_mem, rpe_mem]
    """

    NUM_STATES = 3

    def __init__(self, in_channels: int, embed_dim: int, snn_cfg: Optional[DictConfig] = None):
        super().__init__()
        beta_init = snn_cfg.get('beta_init', 0.5) if snn_cfg else 0.5
        learn_beta = snn_cfg.get('learn_beta', True) if snn_cfg else True
        threshold = snn_cfg.get('threshold', 1.0) if snn_cfg else 1.0
        mid = embed_dim // 2

        self.conv1 = nn.Conv2d(in_channels, mid, 3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid)
        self.lif1 = LIFNeuron(beta_init=beta_init, learn_beta=learn_beta, threshold=threshold)

        self.conv2 = nn.Conv2d(mid, embed_dim, 3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(embed_dim)
        self.lif2 = LIFNeuron(beta_init=beta_init, learn_beta=learn_beta, threshold=threshold)

        self.rpe_conv = nn.Conv2d(embed_dim, embed_dim, 3, padding=1, bias=False)
        self.rpe_bn = nn.BatchNorm2d(embed_dim)
        self.rpe_lif = LIFNeuron(beta_init=beta_init, learn_beta=learn_beta, threshold=threshold)

    def forward(
        self, x: torch.Tensor, prev_mems: Optional[List[Optional[torch.Tensor]]],
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        if prev_mems is None:
            prev_mems = [None] * self.NUM_STATES

        s1, m1 = self.lif1(self.bn1(self.conv1(x)), prev_mems[0])
        s2, m2 = self.lif2(self.bn2(self.conv2(s1)), prev_mems[1])
        rpe_s, rpe_m = self.rpe_lif(self.rpe_bn(self.rpe_conv(s2)), prev_mems[2])
        out = s2 + rpe_s

        return out, [m1, m2, rpe_m]


class SpikingMLP2d(nn.Module):
    """Spiking MLP: Conv1x1 -> BN -> LIF -> Conv1x1 -> BN -> LIF.

    State list (2 tensors): [lif1_mem, lif2_mem]
    """

    NUM_STATES = 2

    def __init__(self, dim: int, mlp_ratio: float = 4.0, snn_cfg: Optional[DictConfig] = None):
        super().__init__()
        beta_init = snn_cfg.get('beta_init', 0.5) if snn_cfg else 0.5
        learn_beta = snn_cfg.get('learn_beta', True) if snn_cfg else True
        threshold = snn_cfg.get('threshold', 1.0) if snn_cfg else 1.0
        hidden = int(dim * mlp_ratio)

        self.fc1 = nn.Conv2d(dim, hidden, 1)
        self.bn1 = nn.BatchNorm2d(hidden)
        self.lif1 = LIFNeuron(beta_init=beta_init, learn_beta=learn_beta, threshold=threshold)

        self.fc2 = nn.Conv2d(hidden, dim, 1)
        self.bn2 = nn.BatchNorm2d(dim)
        self.lif2 = LIFNeuron(beta_init=beta_init, learn_beta=learn_beta, threshold=threshold)

    def forward(
        self, x: torch.Tensor, prev_mems: Optional[List[Optional[torch.Tensor]]],
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        if prev_mems is None:
            prev_mems = [None] * self.NUM_STATES

        s1, m1 = self.lif1(self.bn1(self.fc1(x)), prev_mems[0])
        s2, m2 = self.lif2(self.bn2(self.fc2(s1)), prev_mems[1])
        return s2, [m1, m2]


class SpikingPatchMerging(nn.Module):
    """2x spatial downsample: Conv3x3(C->2C, s=2) + BN + LIF.

    State list (1 tensor): [lif_mem]
    """

    NUM_STATES = 1

    def __init__(self, dim: int, snn_cfg: Optional[DictConfig] = None):
        super().__init__()
        beta_init = snn_cfg.get('beta_init', 0.5) if snn_cfg else 0.5
        learn_beta = snn_cfg.get('learn_beta', True) if snn_cfg else True
        threshold = snn_cfg.get('threshold', 1.0) if snn_cfg else 1.0

        self.conv = nn.Conv2d(dim, 2 * dim, 3, stride=2, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(2 * dim)
        self.lif = LIFNeuron(beta_init=beta_init, learn_beta=learn_beta, threshold=threshold)

    def forward(
        self, x: torch.Tensor, prev_mem: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        spike, mem = self.lif(self.bn(self.conv(x)), prev_mem)
        return spike, mem


class SpikingSwinBlock(nn.Module):
    """Spiking Swin Transformer Block: CGRA Attention + MLP with residuals.

    State list (8 tensors): [attn_mem0..5, mlp_mem0..1]
    """

    ATTN_STATES = CGRAWindowAttention.NUM_STATES  # 6
    MLP_STATES = SpikingMLP2d.NUM_STATES          # 2
    NUM_STATES = ATTN_STATES + MLP_STATES         # 8

    def __init__(
        self,
        dim: int,
        input_resolution: Tuple[int, int],
        num_heads: int,
        window_size: Optional[int] = 8,
        shift_size: int = 0,
        mlp_ratio: float = 4.0,
        attn_scale: float = 0.125,
        output_scale: float = 0.25,
        use_rel_pos_bias: bool = True,
        snn_cfg: Optional[DictConfig] = None,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        H, W = input_resolution

        if window_size is None or min(H, W) <= window_size:
            self.shift_size = 0
            self.global_attn = True
            attn_window = (H, W)
            self.window_size = min(H, W)  # not used for partitioning
        else:
            self.window_size = window_size
            self.shift_size = shift_size
            self.global_attn = False
            attn_window = (window_size, window_size)

        self.attn = CGRAWindowAttention(
            dim=dim, window_size=attn_window, num_heads=num_heads,
            attn_scale=attn_scale, output_scale=output_scale,
            use_rel_pos_bias=use_rel_pos_bias, snn_cfg=snn_cfg,
        )
        self.mlp = SpikingMLP2d(dim=dim, mlp_ratio=mlp_ratio, snn_cfg=snn_cfg)

        if self.shift_size > 0:
            attn_mask = compute_sw_mask(H, W, self.window_size, self.shift_size)
            self.register_buffer('attn_mask', attn_mask)
        else:
            self.attn_mask = None

    def forward(
        self,
        x: torch.Tensor,
        prev_mems: Optional[List[Optional[torch.Tensor]]],
        B: int,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            x: (B, C, H, W)
            prev_mems: list of 8 batch-first membrane tensors, or None.
            B: batch size.
        Returns:
            output: (B, C, H, W)
            new_mems: list of 8 batch-first membrane tensors.
        """
        if prev_mems is None:
            attn_mems = None
            mlp_mems = None
        else:
            attn_mems = prev_mems[:self.ATTN_STATES]
            mlp_mems = prev_mems[self.ATTN_STATES:]

        _, C, H, W = x.shape
        shortcut = x

        # --- Attention ---
        if self.global_attn:
            N = H * W
            x_tokens = x.reshape(B, C, N).permute(0, 2, 1).contiguous()  # (B, N, C)
            attn_out, attn_new = self.attn(x_tokens, attn_mems, B, mask=None)
            x = attn_out.permute(0, 2, 1).contiguous().reshape(B, C, H, W)
        else:
            ws = self.window_size
            x = x.permute(0, 2, 3, 1).contiguous()  # (B, H, W, C)

            if self.shift_size > 0:
                x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))

            windows = window_partition(x, ws)             # (B*nW, ws, ws, C)
            windows = windows.view(-1, ws * ws, C)        # (B*nW, N, C)

            attn_out, attn_new = self.attn(windows, attn_mems, B, mask=self.attn_mask)

            x = attn_out.view(-1, ws, ws, C)
            x = window_reverse(x, ws, H, W)              # (B, H, W, C)

            if self.shift_size > 0:
                x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

            x = x.permute(0, 3, 1, 2).contiguous()       # (B, C, H, W)

        x = shortcut + x

        # --- MLP ---
        mlp_out, mlp_new = self.mlp(x, mlp_mems)
        x = x + mlp_out

        return x, attn_new + mlp_new


class SpikingSwinStage(nn.Module):
    """One stage: N blocks + optional patch merging + optional readout LIF.

    State: flat list of all membrane tensors in the stage.
    """

    def __init__(
        self,
        dim: int,
        input_resolution: Tuple[int, int],
        depth: int,
        num_heads: int,
        window_size: Optional[int] = 8,
        mlp_ratio: float = 4.0,
        attn_scale: float = 0.125,
        output_scale: float = 0.25,
        downsample: bool = True,
        has_readout: bool = True,
        use_rel_pos_bias: bool = True,
        snn_cfg: Optional[DictConfig] = None,
    ):
        super().__init__()
        self.depth = depth
        self.has_downsample = downsample
        self.has_readout = has_readout

        self.blocks = nn.ModuleList()
        for i in range(depth):
            shift = 0 if (i % 2 == 0) else (window_size // 2 if window_size else 0)
            self.blocks.append(SpikingSwinBlock(
                dim=dim, input_resolution=input_resolution, num_heads=num_heads,
                window_size=window_size, shift_size=shift, mlp_ratio=mlp_ratio,
                attn_scale=attn_scale, output_scale=output_scale,
                use_rel_pos_bias=use_rel_pos_bias, snn_cfg=snn_cfg,
            ))

        self.downsample = SpikingPatchMerging(dim, snn_cfg=snn_cfg) if downsample else None

        # Readout path: Conv1x1 + BN + LIF -> membrane for FPN
        # Only created for stages whose output is consumed (e.g. by FPN).
        if has_readout:
            self.readout_conv = nn.Conv2d(dim, dim, 1, bias=False)
            self.readout_bn = nn.BatchNorm2d(dim)
            self.readout_lif = LIFNeuron(
                beta_init=snn_cfg.get('beta_init', 0.5) if snn_cfg else 0.5,
                learn_beta=snn_cfg.get('learn_beta', True) if snn_cfg else True,
                threshold=snn_cfg.get('threshold', 1.0) if snn_cfg else 1.0,
            )

        # Compute state sizes
        self._block_states = SpikingSwinBlock.NUM_STATES  # 8
        self._merge_states = SpikingPatchMerging.NUM_STATES if downsample else 0  # 0 or 1
        self._readout_states = 1 if has_readout else 0
        self.num_states = depth * self._block_states + self._merge_states + self._readout_states

    def forward(
        self,
        x: torch.Tensor,
        prev_state: Optional[SwinSubState],
        B: int,
    ) -> Tuple[Optional[FeatureMap], torch.Tensor, SwinSubState]:
        """
        Args:
            x: (B, C, H, W)
            prev_state: flat list of membrane tensors, or None.
            B: batch size.
        Returns:
            readout_mem: (B, C, H, W) continuous membrane for FPN, or None.
            x_out: (B, C', H', W') output for next stage (spike after merge, or spike).
            new_state: flat list of membrane tensors.
        """
        new_state: List[torch.Tensor] = []
        offset = 0

        # --- Blocks ---
        for i, blk in enumerate(self.blocks):
            if prev_state is not None:
                blk_mems = prev_state[offset:offset + self._block_states]
            else:
                blk_mems = None
            x, blk_new = blk(x, blk_mems, B)
            new_state.extend(blk_new)
            offset += self._block_states

        # --- Readout (membrane for FPN) ---
        if self.has_readout:
            readout_prev = prev_state[offset + self._merge_states] if prev_state is not None else None
            readout_cur = self.readout_bn(self.readout_conv(x))
            _, readout_mem = self.readout_lif(readout_cur, readout_prev)
        else:
            readout_mem = None

        # --- Downsample ---
        if self.downsample is not None:
            merge_prev = prev_state[offset] if prev_state is not None else None
            x, merge_mem = self.downsample(x, merge_prev)
            new_state.append(merge_mem)

        if readout_mem is not None:
            new_state.append(readout_mem)

        return readout_mem, x, new_state


# ---------------------------------------------------------------------------
# Top-level backbone
# ---------------------------------------------------------------------------

class SNNSwinBackbone(BaseDetector):
    """SNN Swin Transformer backbone with CGRA attention.

    Config keys (under model.backbone):
        name: "SNNSwin"
        input_channels: 20
        embed_dim: 64
        depths: [2, 2, 2, 2]
        num_heads: [2, 4, 8, 16]
        window_sizes: [8, 8, 4, null]
        mlp_ratio: 4.0
        attn_scale: 0.125
        output_scale: 0.25
        snn:
            beta_init: 0.5
            learn_beta: true
            threshold: 1.0
            reset_mechanism: subtract
    """

    def __init__(self, mdl_config: DictConfig):
        super().__init__()

        in_channels = mdl_config.input_channels
        embed_dim = mdl_config.embed_dim
        depths = list(mdl_config.depths)
        num_heads = list(mdl_config.num_heads)
        window_sizes = [None if ws is None else int(ws) for ws in mdl_config.window_sizes]
        mlp_ratio = mdl_config.get('mlp_ratio', 4.0)
        attn_scale = mdl_config.get('attn_scale', 0.125)
        output_scale = mdl_config.get('output_scale', 0.25)
        use_rel_pos_bias = mdl_config.get('use_rel_pos_bias', True)
        output_stages = set(mdl_config.get('output_stages', [1, 2, 3, 4]))
        snn_cfg = mdl_config.snn
        in_res_hw = tuple(mdl_config.in_res_hw)

        num_stages = len(depths)
        assert num_stages == 4
        assert len(num_heads) == num_stages
        assert len(window_sizes) == num_stages

        # Patch embedding
        self.patch_embed = SpikingPatchEmbed(
            in_channels=in_channels, embed_dim=embed_dim, snn_cfg=snn_cfg,
        )

        # Compute spatial resolutions
        H, W = in_res_hw[0] // 4, in_res_hw[1] // 4  # after patch embed

        # Build stages
        self.stages = nn.ModuleList()
        self.stage_dims: List[int] = []
        self._strides: List[int] = []
        dim = embed_dim
        stride = 4  # patch embed stride

        for i in range(num_stages):
            self.stages.append(SpikingSwinStage(
                dim=dim, input_resolution=(H, W), depth=depths[i],
                num_heads=num_heads[i], window_size=window_sizes[i],
                mlp_ratio=mlp_ratio, attn_scale=attn_scale, output_scale=output_scale,
                downsample=(i < num_stages - 1),
                has_readout=((i + 1) in output_stages),
                use_rel_pos_bias=use_rel_pos_bias, snn_cfg=snn_cfg,
            ))
            self.stage_dims.append(dim)
            self._strides.append(stride)

            if i < num_stages - 1:
                H, W = H // 2, W // 2
                dim = dim * 2
                stride = stride * 2

        self.num_stages = num_stages

    def get_stage_dims(self, stages: Tuple[int, ...]) -> Tuple[int, ...]:
        stage_indices = [x - 1 for x in stages]
        assert min(stage_indices) >= 0
        assert max(stage_indices) < self.num_stages
        return tuple(self.stage_dims[i] for i in stage_indices)

    def get_strides(self, stages: Tuple[int, ...]) -> Tuple[int, ...]:
        stage_indices = [x - 1 for x in stages]
        assert min(stage_indices) >= 0
        assert max(stage_indices) < self.num_stages
        return tuple(self._strides[i] for i in stage_indices)

    def forward(
        self,
        x: torch.Tensor,
        prev_states: Optional[SwinStates] = None,
        token_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[BackboneFeatures, SwinStates]:
        """
        Args:
            x: (B, C_in, H, W) event representation.
            prev_states: [embed_state, stage0_state, ...] or None.
            token_mask: ignored (interface compatibility).
        Returns:
            features: {1: readout1, 2: readout2, 3: readout3, 4: readout4}
            states: [embed_state, stage0_state, stage1_state, stage2_state, stage3_state]
        """
        B = x.shape[0]

        if prev_states is None:
            prev_states = [None] * (1 + self.num_stages)
        else:
            # Truncated BPTT: detach all membranes from previous timestep graph
            prev_states = RNNStates.recursive_detach(prev_states)

        # Patch embedding
        x, embed_state = self.patch_embed(x, prev_states[0])

        # Stages
        output: Dict[int, FeatureMap] = {}
        states: SwinStates = [embed_state]

        for i, stage in enumerate(self.stages):
            readout_mem, x, stage_state = stage(x, prev_states[i + 1], B)
            if readout_mem is not None:
                output[i + 1] = readout_mem
            states.append(stage_state)

        return output, states
