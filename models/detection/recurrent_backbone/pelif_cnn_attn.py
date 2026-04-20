"""PeLIF-CNN backbone with spike-driven attention.

Extension of PeLIFCNN (models/detection/recurrent_backbone/pelif_cnn.py) that
adds RVT-style window + grid spike attention at the end of each stage,
followed by a readout PeLIF neuron.

Architecture per stage:
    PeLIFConvBlock (downsample, recurrent)
    PeLIFConvBlock (refine, recurrent)   × (N-1)
    ↓ spike (binary, local spatial + temporal features)
    SpikePartitionAttention (window) + residual
    SpikePartitionAttention (grid)   + residual
    ↓ attn_out (float, coincidence-weighted)
    PeLIFNeuron2d (readout)
    ├→ membrane → FPN
    └→ spike → next stage

Key properties:
    - Attention is spike-driven: Q, K, V are binary spikes.
    - Q @ K^T is coincidence count (integer), no softmax.
    - Readout PeLIF neuron converts float attention output back to spike + membrane.
    - Every stage output format matches PeLIFCNN: (membrane to FPN, spike to next stage).
"""
from typing import Dict, List, Optional, Tuple

import torch as th
import torch.nn as nn
from omegaconf import DictConfig

from data.utils.types import FeatureMap, BackboneFeatures
from models.layers.pelif_spiking import PeLIFNeuron2d
from models.layers.spike_attention import SpikePartitionAttention
from models.layers.spiking import LIFNeuron
from models.layers.maxvit.maxvit import PartitionType
from .base import BaseDetector
from .pelif_cnn import PeLIFConvBlock


# State types
LayerState = Optional[Tuple[th.Tensor, th.Tensor]]
# Attention state: (q_mem, k_mem, v_mem) each (N, H, W, C) — unpartitioned
AttnState = Optional[Tuple[th.Tensor, th.Tensor, th.Tensor]]
# Post-attention LIF state: single membrane tensor (N, H, W, C)
PostLIFState = Optional[th.Tensor]
ReadoutState = Optional[Tuple[th.Tensor, th.Tensor]]
# StageState = (conv_states,
#               window_attn_state, post_window_lif_mem,
#               grid_attn_state,   post_grid_lif_mem,
#               readout_state)
StageState = Tuple[List[LayerState], AttnState, PostLIFState,
                   AttnState, PostLIFState, ReadoutState]
PeLIFAttnStates = List[StageState]


class PeLIFCNNAttnStage(nn.Module):
    """Stage = stacked PeLIFConvBlocks + window/grid spike attention + readout neuron.

    Args:
        dim_in, dim_out: channels
        spatial_downsample_factor: stride for first layer
        num_layers: number of PeLIFConvBlocks before attention
        periods: clock periods for PeLIF neurons
        pelif_cfg: DictConfig for PeLIF hyperparams
        rec_kernel_size: recurrent conv kernel (1 or 3)
        norm: 'bn' or 'none' for PeLIFConvBlock
        partition_size: (h, w) for window/grid attention
        dim_head: per-head dim for spike attention
        attn_threshold: LIF threshold inside attention
    """

    def __init__(self,
                 dim_in: int,
                 dim_out: int,
                 spatial_downsample_factor: int,
                 num_layers: int,
                 periods: Tuple[int, ...],
                 pelif_cfg: DictConfig,
                 rec_kernel_size: int = 1,
                 norm: str = 'none',
                 partition_size: Tuple[int, int] = (4, 5),
                 dim_head: int = 32,
                 attn_threshold: float = 1.0,
                 attn_beta_init: float = 0.9,
                 attn_learn_beta: bool = True,
                 attn_qkv_bias: bool = False):
        super().__init__()
        self.num_layers = num_layers

        pelif_kwargs = dict(
            periods=periods,
            rec_kernel_size=rec_kernel_size,
            beta_init=pelif_cfg.get('beta_init', 0.9),
            learn_beta=pelif_cfg.get('learn_beta', True),
            threshold=pelif_cfg.get('threshold', 1.0),
            surrogate=pelif_cfg.get('surrogate', 'triangle'),
            norm=norm,
        )

        # First conv block: downsample
        if spatial_downsample_factor == 4:
            first_kernel = 7
        else:
            first_kernel = 3

        layers = [PeLIFConvBlock(
            in_channels=dim_in,
            out_channels=dim_out,
            kernel_size=first_kernel,
            stride=spatial_downsample_factor,
            **pelif_kwargs,
        )]

        # Remaining refine layers (stride=1)
        for _ in range(num_layers - 1):
            layers.append(PeLIFConvBlock(
                in_channels=dim_out,
                out_channels=dim_out,
                kernel_size=3,
                stride=1,
                **pelif_kwargs,
            ))

        self.layers = nn.ModuleList(layers)

        # Spike attention: window + grid (RVT-style pair)
        attn_kwargs = dict(
            dim=dim_out,
            partition_size=partition_size,
            dim_head=dim_head,
            qkv_bias=attn_qkv_bias,
            beta_init=attn_beta_init,
            learn_beta=attn_learn_beta,
            threshold=attn_threshold,
            surrogate=pelif_cfg.get('surrogate', 'triangle'),
        )
        self.attn_window = SpikePartitionAttention(
            partition_type=PartitionType.WINDOW, **attn_kwargs)
        self.attn_grid = SpikePartitionAttention(
            partition_type=PartitionType.GRID, **attn_kwargs)

        # Post-attention LIF: rebinarize float (residual + attn output) back to spike
        # so that downstream modules (grid attn, readout) receive binary input.
        # Maintains "spike-only between modules" principle.
        post_lif_kwargs = dict(
            beta_init=attn_beta_init,
            learn_beta=attn_learn_beta,
            threshold=attn_threshold,
            reset_mechanism='subtract',
            channels=None,
            surrogate=pelif_cfg.get('surrogate', 'triangle'),
        )
        self.lif_post_window = LIFNeuron(**post_lif_kwargs)
        self.lif_post_grid = LIFNeuron(**post_lif_kwargs)

        # Readout PeLIF neuron: converts attention output (float) back to
        # spike + membrane. Receives the post-attention feature as input current.
        self.readout = PeLIFNeuron2d(
            channels=dim_out,
            periods=periods,
            beta_init=pelif_cfg.get('beta_init', 0.9),
            learn_beta=pelif_cfg.get('learn_beta', True),
            threshold=pelif_cfg.get('threshold', 1.0),
            surrogate=pelif_cfg.get('surrogate', 'triangle'),
        )

    def forward(self,
                x: th.Tensor,
                prev_state: Optional[StageState] = None,
                t: int = 0,
                ) -> Tuple[FeatureMap, th.Tensor, StageState]:
        """
        Args:
            x: (N, C_in, H, W) input (spike or raw events at stage 0)
            prev_state: (conv_states, window_attn_state, grid_attn_state, readout_state)
            t: current timestep

        Returns:
            membrane: (N, C_out, H', W') from readout neuron → FPN
            spike:    (N, C_out, H', W') from readout neuron → next stage
            new_state: full stage state tuple
        """
        if prev_state is None:
            conv_states = [None] * self.num_layers
            attn_window_state = None
            post_window_lif_mem = None
            attn_grid_state = None
            post_grid_lif_mem = None
            readout_state = None
        else:
            (conv_states, attn_window_state, post_window_lif_mem,
             attn_grid_state, post_grid_lif_mem, readout_state) = prev_state

        # PeLIF conv stack (recurrent, spike output)
        new_conv_states = []
        for i, layer in enumerate(self.layers):
            spike, mem, ls = layer(x, conv_states[i], t)
            new_conv_states.append(ls)
            x = spike

        # spike: (N, C, H, W) from last conv block → channel-last for attention
        spike_cl = x.permute(0, 2, 3, 1).contiguous()   # (N, H, W, C) binary

        # --- Window attention block: attn + residual + LIF rebinarize ---
        attn_w_out, new_attn_window_state = self.attn_window(spike_cl, attn_window_state)
        feat_cl = spike_cl + attn_w_out                                # binary + float = float
        spike_cl, new_post_window_lif_mem = self.lif_post_window(
            feat_cl, post_window_lif_mem)                              # float → binary spike

        # --- Grid attention block: attn + residual + LIF rebinarize ---
        # Grid attn now receives binary spike (spike-only between modules).
        attn_g_out, new_attn_grid_state = self.attn_grid(spike_cl, attn_grid_state)
        feat_cl = spike_cl + attn_g_out                                # binary + float = float
        spike_cl, new_post_grid_lif_mem = self.lif_post_grid(
            feat_cl, post_grid_lif_mem)                                # float → binary spike

        # Back to channel-first for readout neuron
        spike_cf = spike_cl.permute(0, 3, 1, 2).contiguous()           # (N, C, H, W) binary

        # Readout: binary spike input current → spike + membrane via PeLIF dynamics
        if readout_state is not None:
            readout_mem_prev, _ = readout_state
        else:
            readout_mem_prev = None

        readout_spike, readout_mem = self.readout(spike_cf, readout_mem_prev, t)
        new_readout_state = (readout_mem, readout_spike)

        return (readout_mem, readout_spike,
                (new_conv_states,
                 new_attn_window_state, new_post_window_lif_mem,
                 new_attn_grid_state, new_post_grid_lif_mem,
                 new_readout_state))


class PeLIFCNNAttnBackbone(BaseDetector):
    """PeLIF-CNN + spike attention backbone (4 stages).

    Config keys (under model.backbone):
        name: "PeLIFCNNAttn"
        input_channels: 20
        embed_dim: 64
        dim_multiplier: [1, 2, 4, 8]
        num_conv_layers: [2, 2, 2, 2]
        periods: [1, 2, 4, 8]
        rec_kernel_size: 3
        norm: 'none'
        stem:
            patch_size: 4
        pelif:
            beta_init: 0.9
            learn_beta: true
            threshold: 1.0
            surrogate: triangle
        attention:
            partition_size: [4, 5]
            dim_head: 32
            threshold: 1.0
            qkv_bias: false
    """

    def __init__(self, mdl_config: DictConfig):
        super().__init__()

        in_channels = mdl_config.input_channels
        embed_dim = mdl_config.embed_dim
        dim_multiplier = tuple(mdl_config.dim_multiplier)
        num_conv_layers_per_stage = tuple(mdl_config.num_conv_layers)
        patch_size = mdl_config.stem.patch_size
        periods = tuple(mdl_config.periods)
        pelif_cfg = mdl_config.pelif
        rec_kernel_size = mdl_config.get('rec_kernel_size', 1)
        norm = mdl_config.get('norm', 'none')

        attn_cfg = mdl_config.get('attention', {})
        partition_size = tuple(attn_cfg.get('partition_size', (4, 5)))
        dim_head = attn_cfg.get('dim_head', 32)
        attn_threshold = attn_cfg.get('threshold', 1.0)
        attn_beta_init = attn_cfg.get('beta_init', 0.9)
        attn_learn_beta = attn_cfg.get('learn_beta', True)
        attn_qkv_bias = attn_cfg.get('qkv_bias', False)

        num_stages = len(dim_multiplier)
        assert num_stages == 4

        self.stage_dims = [embed_dim * m for m in dim_multiplier]
        self.stages = nn.ModuleList()
        self._strides = []

        input_dim = in_channels
        stride = 1
        for stage_idx in range(num_stages):
            ds_factor = patch_size if stage_idx == 0 else 2
            stage_dim = self.stage_dims[stage_idx]

            self.stages.append(PeLIFCNNAttnStage(
                dim_in=input_dim,
                dim_out=stage_dim,
                spatial_downsample_factor=ds_factor,
                num_layers=num_conv_layers_per_stage[stage_idx],
                periods=periods,
                pelif_cfg=pelif_cfg,
                rec_kernel_size=rec_kernel_size,
                norm=norm,
                partition_size=partition_size,
                dim_head=dim_head,
                attn_threshold=attn_threshold,
                attn_beta_init=attn_beta_init,
                attn_learn_beta=attn_learn_beta,
                attn_qkv_bias=attn_qkv_bias,
            ))

            stride *= ds_factor
            self._strides.append(stride)
            input_dim = stage_dim

        self.num_stages = num_stages
        self._timestep = 0

    def get_stage_dims(self, stages: Tuple[int, ...]) -> Tuple[int, ...]:
        stage_indices = [x - 1 for x in stages]
        return tuple(self.stage_dims[i] for i in stage_indices)

    def get_strides(self, stages: Tuple[int, ...]) -> Tuple[int, ...]:
        stage_indices = [x - 1 for x in stages]
        return tuple(self._strides[i] for i in stage_indices)

    def forward(self,
                x: th.Tensor,
                prev_states: Optional[PeLIFAttnStates] = None,
                token_mask: Optional[th.Tensor] = None,
                ) -> Tuple[BackboneFeatures, PeLIFAttnStates]:
        if prev_states is None:
            prev_states = [None] * self.num_stages
            self._timestep = 0

        t = self._timestep

        output: Dict[int, FeatureMap] = {}
        states: PeLIFAttnStates = []

        for stage_idx, stage in enumerate(self.stages):
            membrane, spike, stage_state = stage(x, prev_states[stage_idx], t)
            states.append(stage_state)
            output[stage_idx + 1] = membrane
            x = spike

        self._timestep += 1
        return output, states
