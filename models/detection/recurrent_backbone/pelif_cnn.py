"""PeLIF-CNN backbone: LIF for spatial extraction + PeLIF for temporal memory.

Architecture per stage:
    [LIF Conv layers] → spatial feature extraction (responds to every input)
    [PeLIF block]     → temporal memory with multi-period recurrence

This separation ensures:
    - Events are processed immediately by LIF (no period gating on features)
    - Temporal memory is maintained by PeLIF even when events are absent
"""
from typing import Dict, List, Optional, Tuple

import torch as th
import torch.nn as nn
from omegaconf import DictConfig

from data.utils.types import FeatureMap, BackboneFeatures
from models.layers.spiking import SpikingConvBlock
from models.layers.pelif_spiking import PeLIFNeuron2d
from .base import BaseDetector

# State types
LIFState = Optional[th.Tensor]                               # membrane per conv layer
PeLIFBlockState = Optional[Tuple[th.Tensor, th.Tensor]]      # (mem, prev_spike)
StageState = Tuple[List[LIFState], PeLIFBlockState]           # (lif_mems, pelif_state)
PeLIFStates = List[StageState]


class PeLIFTemporalBlock(nn.Module):
    """PeLIF temporal memory block with 1x1 Conv recurrence.

    Placed at the end of each stage. Receives LIF features,
    applies PeLIF dynamics for multi-timescale temporal memory.
    """

    def __init__(self,
                 channels: int,
                 periods: Tuple[int, ...] = (1, 2, 4, 8),
                 beta_init: float = 0.9,
                 learn_beta: bool = True,
                 threshold: float = 1.0,
                 surrogate: str = 'triangle'):
        super().__init__()
        self.pelif = PeLIFNeuron2d(
            channels=channels,
            periods=periods,
            beta_init=beta_init,
            learn_beta=learn_beta,
            surrogate=surrogate,
            threshold=threshold,
        )
        # 1x1 Conv recurrence (W_rec equivalent)
        self.conv_rec = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        nn.init.orthogonal_(self.conv_rec.weight.view(channels, -1))
        self.conv_rec.weight.data *= 0.5

    def forward(self,
                x: th.Tensor,
                state: PeLIFBlockState = None,
                t: int = 0,
                ) -> Tuple[th.Tensor, th.Tensor, PeLIFBlockState]:
        """
        Args:
            x: (N, C, H, W) input from LIF conv layers
            state: (mem, prev_spike) or None
            t: current timestep
        Returns:
            spike: (N, C, H, W) binary
            mem: (N, C, H, W) continuous (for FPN)
            new_state: (mem, spike)
        """
        if state is not None:
            mem, prev_spike = state
        else:
            mem, prev_spike = None, None

        cur = x
        if prev_spike is not None:
            cur = cur + self.conv_rec(prev_spike)

        spike, mem = self.pelif(cur, mem, t)
        return spike, mem, (mem, spike)


class PeLIFCNNStage(nn.Module):
    """One stage: LIF conv layers (spatial) + PeLIF block (temporal).

    Architecture:
        Conv2d+BN+LIF (downsample) → Conv2d+BN+LIF (refine) → PeLIF temporal block
    """

    def __init__(self,
                 dim_in: int,
                 dim_out: int,
                 spatial_downsample_factor: int,
                 num_conv_layers: int,
                 periods: Tuple[int, ...],
                 snn_cfg: DictConfig,
                 pelif_cfg: DictConfig):
        super().__init__()
        self.num_conv_layers = num_conv_layers

        # SNN (LIF) kwargs for spatial feature extraction
        snn_kwargs = dict(
            beta_init=snn_cfg.get('beta_init', 0.9),
            learn_beta=snn_cfg.get('learn_beta', True),
            threshold=snn_cfg.get('threshold', 1.0),
            reset_mechanism=snn_cfg.get('reset_mechanism', 'subtract'),
            channelwise_beta=snn_cfg.get('channelwise_beta', False),
            beta_spread=snn_cfg.get('beta_spread', 0.0),
        )

        # LIF conv layers (spatial feature extraction)
        if spatial_downsample_factor == 4:
            k, p = 7, 3
        else:
            k, p = 3, 1

        lif_layers = [SpikingConvBlock(
            in_channels=dim_in,
            out_channels=dim_out,
            kernel_size=k,
            stride=spatial_downsample_factor,
            padding=p,
            **snn_kwargs,
        )]
        for _ in range(num_conv_layers - 1):
            lif_layers.append(SpikingConvBlock(
                in_channels=dim_out,
                out_channels=dim_out,
                kernel_size=3,
                stride=1,
                padding=1,
                **snn_kwargs,
            ))
        self.lif_layers = nn.ModuleList(lif_layers)

        # PeLIF temporal memory block (at stage end)
        self.pelif_block = PeLIFTemporalBlock(
            channels=dim_out,
            periods=periods,
            beta_init=pelif_cfg.get('beta_init', 0.9),
            surrogate=pelif_cfg.get('surrogate', 'triangle'),
            learn_beta=pelif_cfg.get('learn_beta', True),
            threshold=pelif_cfg.get('threshold', 1.0),
        )

    def forward(self,
                x: th.Tensor,
                prev_state: Optional[StageState] = None,
                t: int = 0,
                ) -> Tuple[FeatureMap, th.Tensor, StageState]:
        """
        Returns:
            membrane: continuous feature for FPN (from PeLIF)
            spike: binary for next stage (from PeLIF)
            new_state: (lif_mems, pelif_state)
        """
        if prev_state is not None:
            lif_mems, pelif_state = prev_state
        else:
            lif_mems = [None] * self.num_conv_layers
            pelif_state = None

        # LIF layers: spatial feature extraction
        new_lif_mems = []
        for i, layer in enumerate(self.lif_layers):
            spike, mem = layer(x, lif_mems[i])
            new_lif_mems.append(mem)
            x = spike  # spike feeds next LIF layer

        # PeLIF block: temporal memory (receives LIF spike output)
        pelif_spike, pelif_mem, new_pelif_state = self.pelif_block(x, pelif_state, t)

        return pelif_mem.clone(), pelif_spike, (new_lif_mems, new_pelif_state)


class PeLIFCNNBackbone(BaseDetector):
    """PeLIF-CNN backbone: LIF spatial + PeLIF temporal, 4 stages.

    Config keys (under model.backbone):
        name: "PeLIFCNN"
        input_channels: 20
        embed_dim: 64
        dim_multiplier: [1, 2, 4, 8]
        num_conv_layers: [2, 2, 2, 2]
        periods: [1, 2, 4, 8]
        stem:
            patch_size: 4
        snn:
            beta_init: 0.9
            learn_beta: true
            threshold: 1.0
            reset_mechanism: subtract
        pelif:
            beta_init: 0.9
            learn_beta: true
            threshold: 1.0
    """

    def __init__(self, mdl_config: DictConfig):
        super().__init__()

        in_channels = mdl_config.input_channels
        embed_dim = mdl_config.embed_dim
        dim_multiplier = tuple(mdl_config.dim_multiplier)
        num_conv_layers_per_stage = tuple(mdl_config.num_conv_layers)
        patch_size = mdl_config.stem.patch_size
        periods = tuple(mdl_config.periods)
        snn_cfg = mdl_config.snn
        pelif_cfg = mdl_config.pelif

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

            self.stages.append(PeLIFCNNStage(
                dim_in=input_dim,
                dim_out=stage_dim,
                spatial_downsample_factor=ds_factor,
                num_conv_layers=num_conv_layers_per_stage[stage_idx],
                periods=periods,
                snn_cfg=snn_cfg,
                pelif_cfg=pelif_cfg,
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
                prev_states: Optional[PeLIFStates] = None,
                token_mask: Optional[th.Tensor] = None,
                ) -> Tuple[BackboneFeatures, PeLIFStates]:
        if prev_states is None:
            prev_states = [None] * self.num_stages
            self._timestep = 0

        t = self._timestep

        output: Dict[int, FeatureMap] = {}
        states: PeLIFStates = []

        for stage_idx, stage in enumerate(self.stages):
            membrane, spike, stage_state = stage(x, prev_states[stage_idx], t)
            states.append(stage_state)
            output[stage_idx + 1] = membrane
            x = spike

        self._timestep += 1
        return output, states
