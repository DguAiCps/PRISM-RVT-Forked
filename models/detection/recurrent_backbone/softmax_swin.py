"""SoftmaxSwin Backbone: Ablation replacing SSA with standard softmax attention.

All spiking components (patch embed, MLP, downsample, readout) are preserved.
Only the attention Q/K/V projections and attention computation are changed to
standard scaled dot-product attention with softmax normalization.

This isolates one variable: unnormalized spike attention vs. softmax attention.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from timm.layers import trunc_normal_

from data.utils.types import BackboneFeatures, FeatureMap
from models.layers.spiking import LIFNeuron
from modules.utils.detection import RNNStates
from .base import BaseDetector
from .snn_swin import (
    SwinSubState,
    SwinStates,
    window_partition,
    window_reverse,
    compute_sw_mask,
    SpikingPatchEmbed,
    SpikingMLP2d,
    SpikingPatchMerging,
)


# ---------------------------------------------------------------------------
# Softmax Window Attention
# ---------------------------------------------------------------------------

class SoftmaxWindowAttention(nn.Module):
    """Standard scaled dot-product attention with softmax normalization.

    Stateless (NUM_STATES = 0) — no LIF membranes.
    Drop-in replacement for SSAWindowAttention with identical forward signature.

    State list: [] (empty)
    """

    NUM_STATES = 0

    def __init__(
        self,
        dim: int,
        window_size: Tuple[int, int],
        num_heads: int,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        use_rel_pos_bias: bool = True,
        snn_cfg: Optional[DictConfig] = None,  # accepted but unused (interface compat)
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.N = window_size[0] * window_size[1]
        self.use_rel_pos_bias = use_rel_pos_bias

        # Q/K/V: single fused linear projection
        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(attn_drop)

        # Output projection
        self.proj_out = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Relative position bias (optional, same as SSA)
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
            prev_mems: ignored (stateless).
            B: batch size.
            mask: (nW, N, N) shifted-window mask or None.
        Returns:
            output: (B*nW, N, C)
            new_mems: [] (empty list)
        """
        BnW, N, C = x.shape
        H_heads, d = self.num_heads, self.head_dim

        # Q/K/V projection
        qkv = self.qkv(x).reshape(BnW, N, 3, H_heads, d).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # each (BnW, H_heads, N, d)

        # Scaled dot-product
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (BnW, H, N, N)

        # Relative position bias
        if self.use_rel_pos_bias:
            attn = attn + self._get_rel_pos_bias().unsqueeze(0)

        # Shifted-window mask: additive (-100 for invalid) before softmax
        if mask is not None:
            nW_mask = mask.shape[0]
            attn = attn.view(-1, nW_mask, H_heads, N, N)
            attn = attn + mask.unsqueeze(0).unsqueeze(2)  # (1, nW, 1, N, N)
            attn = attn.reshape(-1, H_heads, N, N)

        # Softmax normalization
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        # Store for visualization
        self._last_attn = attn.detach()
        self._last_attn_meta = {'B': B, 'nW': BnW // B, 'N': N, 'H_heads': H_heads}

        # Weighted sum + output projection
        out = (attn @ v).transpose(1, 2).reshape(BnW, N, C)
        out = self.proj_drop(self.proj_out(out))

        return out, []


# ---------------------------------------------------------------------------
# Block
# ---------------------------------------------------------------------------

class SoftmaxSwinBlock(nn.Module):
    """Swin block with softmax attention + spiking MLP.

    State list (2 or 4 tensors):
        Without feedback: [mlp_mem0, mlp_mem1]
        With feedback:    [mlp_mem0, mlp_mem1, fb_spike, fb_mem]
    """

    ATTN_STATES = SoftmaxWindowAttention.NUM_STATES   # 0
    MLP_STATES = SpikingMLP2d.NUM_STATES              # 2

    def __init__(
        self,
        dim: int,
        input_resolution: Tuple[int, int],
        num_heads: int,
        window_size: Optional[int] = 8,
        shift_size: int = 0,
        mlp_ratio: float = 4.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        use_rel_pos_bias: bool = True,
        snn_cfg: Optional[DictConfig] = None,
        use_feedback: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        H, W = input_resolution

        if window_size is None or min(H, W) <= window_size:
            self.shift_size = 0
            self.global_attn = True
            attn_window = (H, W)
            self.window_size = min(H, W)
        else:
            self.window_size = window_size
            self.shift_size = shift_size
            self.global_attn = False
            attn_window = (window_size, window_size)

        self.attn = SoftmaxWindowAttention(
            dim=dim, window_size=attn_window, num_heads=num_heads,
            attn_drop=attn_drop, proj_drop=proj_drop,
            use_rel_pos_bias=use_rel_pos_bias, snn_cfg=snn_cfg,
        )
        self.mlp = SpikingMLP2d(dim=dim, mlp_ratio=mlp_ratio, snn_cfg=snn_cfg)

        if self.shift_size > 0:
            attn_mask = compute_sw_mask(H, W, self.window_size, self.shift_size)
            self.register_buffer('attn_mask', attn_mask)
        else:
            self.attn_mask = None

        # Temporal feedback via LIF-accumulated spike
        self.use_feedback = use_feedback
        if use_feedback:
            self.fb_lif = LIFNeuron(
                beta_init=snn_cfg.get('fb_beta_init', 0.9) if snn_cfg else 0.9,
                learn_beta=True,
                threshold=snn_cfg.get('fb_threshold', 0.25) if snn_cfg else 0.25,
                reset_mechanism='subtract',
            )
            self.FB_STATES = 2  # fb_spike + fb_mem
        else:
            self.FB_STATES = 0
        self.NUM_STATES = self.ATTN_STATES + self.MLP_STATES + self.FB_STATES

    def forward(
        self,
        x: torch.Tensor,
        prev_mems: Optional[List[Optional[torch.Tensor]]],
        B: int,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            x: (B, C, H, W)
            prev_mems: list of NUM_STATES batch-first membrane tensors, or None.
            B: batch size.
        Returns:
            output: (B, C, H, W)
            new_mems: list of NUM_STATES batch-first membrane tensors.
        """
        if prev_mems is None:
            mlp_mems = None
            fb_prev_spike, fb_prev_mem = None, None
        else:
            mlp_mems = prev_mems[self.ATTN_STATES:self.ATTN_STATES + self.MLP_STATES]
            if self.use_feedback:
                fb_prev_spike = prev_mems[self.ATTN_STATES + self.MLP_STATES]
                fb_prev_mem = prev_mems[self.ATTN_STATES + self.MLP_STATES + 1]
            else:
                fb_prev_spike, fb_prev_mem = None, None

        # Inject temporal feedback (binary spike from previous timestep)
        if fb_prev_spike is not None:
            x = x + fb_prev_spike

        _, C, H, W = x.shape
        shortcut = x

        # --- Attention ---
        if self.global_attn:
            N = H * W
            x_tokens = x.reshape(B, C, N).permute(0, 2, 1).contiguous()  # (B, N, C)
            attn_out, attn_new = self.attn(x_tokens, None, B, mask=None)
            x = attn_out.permute(0, 2, 1).contiguous().reshape(B, C, H, W)
        else:
            ws = self.window_size
            x = x.permute(0, 2, 3, 1).contiguous()  # (B, H, W, C)

            if self.shift_size > 0:
                x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))

            windows = window_partition(x, ws)             # (B*nW, ws, ws, C)
            windows = windows.view(-1, ws * ws, C)        # (B*nW, N, C)

            attn_out, attn_new = self.attn(windows, None, B, mask=self.attn_mask)

            x = attn_out.view(-1, ws, ws, C)
            x = window_reverse(x, ws, H, W)              # (B, H, W, C)

            if self.shift_size > 0:
                x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

            x = x.permute(0, 3, 1, 2).contiguous()       # (B, C, H, W)

        # Capture attention output for feedback BEFORE residual
        if self.use_feedback:
            fb_spike, fb_mem = self.fb_lif(x, fb_prev_mem)

        x = shortcut + x

        # --- MLP ---
        mlp_out, mlp_new = self.mlp(x, mlp_mems)
        x = x + mlp_out

        new_mems = attn_new + mlp_new
        if self.use_feedback:
            new_mems.extend([fb_spike, fb_mem])

        return x, new_mems


# ---------------------------------------------------------------------------
# Stage
# ---------------------------------------------------------------------------

class SoftmaxSwinStage(nn.Module):
    """One stage: N softmax-attn blocks + optional patch merging + optional readout LIF.

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
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        downsample: bool = True,
        has_readout: bool = True,
        use_rel_pos_bias: bool = True,
        snn_cfg: Optional[DictConfig] = None,
        use_feedback: bool = False,
    ):
        super().__init__()
        self.depth = depth
        self.has_downsample = downsample
        self.has_readout = has_readout

        self.blocks = nn.ModuleList()
        for i in range(depth):
            shift = 0 if (i % 2 == 0) else (window_size // 2 if window_size else 0)
            self.blocks.append(SoftmaxSwinBlock(
                dim=dim, input_resolution=input_resolution, num_heads=num_heads,
                window_size=window_size, shift_size=shift, mlp_ratio=mlp_ratio,
                attn_drop=attn_drop, proj_drop=proj_drop,
                use_rel_pos_bias=use_rel_pos_bias, snn_cfg=snn_cfg,
                use_feedback=use_feedback,
            ))

        self.downsample = SpikingPatchMerging(dim, snn_cfg=snn_cfg) if downsample else None

        if has_readout:
            self.readout_conv = nn.Conv2d(dim, dim, 1, bias=False)
            self.readout_bn = nn.BatchNorm2d(dim)
            self.readout_lif = LIFNeuron(
                beta_init=snn_cfg.get('beta_init', 0.5) if snn_cfg else 0.5,
                learn_beta=snn_cfg.get('learn_beta', True) if snn_cfg else True,
                threshold=snn_cfg.get('threshold', 1.0) if snn_cfg else 1.0,
            )

        # Use instance NUM_STATES from block (varies with use_feedback)
        self._block_states = self.blocks[0].NUM_STATES
        self._merge_states = SpikingPatchMerging.NUM_STATES if downsample else 0
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
            x_out: (B, C', H', W') output for next stage.
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
# Backbone
# ---------------------------------------------------------------------------

class SoftmaxSwinBackbone(BaseDetector):
    """SoftmaxSwin backbone: softmax attention + spiking everything else.

    Config keys (under model.backbone):
        name: "SoftmaxSwin"
        input_channels: 20
        embed_dim: 64
        depths: [2, 2, 2, 2]
        num_heads: [2, 4, 8, 16]
        window_sizes: [8, 8, 4, null]
        mlp_ratio: 4.0
        attn_drop: 0.0
        proj_drop: 0.0
        use_rel_pos_bias: true
        output_stages: [2, 3, 4]
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
        attn_drop = mdl_config.get('attn_drop', 0.0)
        proj_drop = mdl_config.get('proj_drop', 0.0)
        use_rel_pos_bias = mdl_config.get('use_rel_pos_bias', True)
        output_stages = set(mdl_config.get('output_stages', [1, 2, 3, 4]))
        snn_cfg = mdl_config.snn
        use_feedback = snn_cfg.get('use_feedback', False)
        in_res_hw = tuple(mdl_config.in_res_hw)

        num_stages = len(depths)
        assert num_stages == 4
        assert len(num_heads) == num_stages
        assert len(window_sizes) == num_stages

        # Patch embedding (spiking)
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
        stride = 4

        for i in range(num_stages):
            self.stages.append(SoftmaxSwinStage(
                dim=dim, input_resolution=(H, W), depth=depths[i],
                num_heads=num_heads[i], window_size=window_sizes[i],
                mlp_ratio=mlp_ratio, attn_drop=attn_drop, proj_drop=proj_drop,
                downsample=(i < num_stages - 1),
                has_readout=((i + 1) in output_stages),
                use_rel_pos_bias=use_rel_pos_bias, snn_cfg=snn_cfg,
                use_feedback=use_feedback,
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
            features: {1: readout1, 2: readout2, ...}
            states: [embed_state, stage0_state, ...]
        """
        B = x.shape[0]

        if prev_states is None:
            prev_states = [None] * (1 + self.num_stages)

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
