"""SNN-CNN backbone: drop-in replacement for MaxViTRNN (RNNDetector)."""
from typing import Dict, List, Optional, Tuple

import torch as th
import torch.nn as nn
from omegaconf import DictConfig

from data.utils.types import FeatureMap, BackboneFeatures
from models.layers.spiking import SpikingConvBlock
from .base import BaseDetector

# Type aliases for SNN states
SpikingState = List[Optional[th.Tensor]]   # membrane potentials within one stage
SpikingStates = List[SpikingState]         # all stages


class SNNCNNStage(nn.Module):
    """One stage of the SNN-CNN backbone.

    Architecture per stage:
        Conv2d(stride) + BN + LIF   (downsample + first spiking layer)
        Conv2d(1)      + BN + LIF   (additional spiking layers, same spatial dims)

    Input format:  NCHW (float)
    Output: membrane potential (continuous, for FPN), spike (binary, for next stage)
    """

    def __init__(self,
                 dim_in: int,
                 dim_out: int,
                 spatial_downsample_factor: int,
                 num_conv_layers: int,
                 snn_cfg: DictConfig):
        super().__init__()
        self.num_conv_layers = num_conv_layers

        beta_init = snn_cfg.get('beta_init', 0.9)
        learn_beta = snn_cfg.get('learn_beta', True)
        threshold = snn_cfg.get('threshold', 1.0)
        reset_mechanism = snn_cfg.get('reset_mechanism', 'subtract')

        # First layer: spatial downsampling
        if spatial_downsample_factor == 4:
            k, p = 7, 3  # stem: 7x7 conv, stride 4
        else:
            k, p = 3, 1  # stages 1-3: 3x3 conv, stride 2

        layers = [SpikingConvBlock(
            in_channels=dim_in,
            out_channels=dim_out,
            kernel_size=k,
            stride=spatial_downsample_factor,
            padding=p,
            beta_init=beta_init,
            learn_beta=learn_beta,
            threshold=threshold,
            reset_mechanism=reset_mechanism,
        )]

        # Additional same-resolution layers
        for _ in range(num_conv_layers - 1):
            layers.append(SpikingConvBlock(
                in_channels=dim_out,
                out_channels=dim_out,
                kernel_size=3,
                stride=1,
                padding=1,
                beta_init=beta_init,
                learn_beta=learn_beta,
                threshold=threshold,
                reset_mechanism=reset_mechanism,
            ))

        self.layers = nn.ModuleList(layers)

    def forward(self,
                x: th.Tensor,
                prev_mems: Optional[SpikingState] = None,
                ) -> Tuple[FeatureMap, th.Tensor, SpikingState]:
        """
        Args:
            x: (N, C_in, H, W) input tensor
            prev_mems: list of membrane potentials per spiking layer, or None
        Returns:
            membrane: (N, C_out, H', W') membrane potential of last layer (for FPN)
            spike: (N, C_out, H', W') spike output of last layer (for next stage)
            new_mems: list of new membrane potentials
        """
        if prev_mems is None:
            prev_mems = [None] * self.num_conv_layers

        new_mems = []
        for i, layer in enumerate(self.layers):
            spike, mem = layer(x, prev_mems[i])
            new_mems.append(mem)
            x = spike  # spike feeds into next layer

        # Clone membrane for feature output to avoid in-place modification
        # by snntorch's reset mechanism during the next timestep.
        return mem.clone(), spike, new_mems


class SNNCNNBackbone(BaseDetector):
    """SNN-CNN backbone with 4 stages, matching RNNDetector interface.

    Config keys (under model.backbone):
        name: "SNNCNN"
        input_channels: 20
        embed_dim: 64
        dim_multiplier: [1, 2, 4, 8]
        num_conv_layers: [2, 2, 2, 2]
        stem:
            patch_size: 4
        snn:
            beta_init: 0.9
            learn_beta: true
            threshold: 1.0
            reset_mechanism: "subtract"
    """

    def __init__(self, mdl_config: DictConfig):
        super().__init__()

        in_channels = mdl_config.input_channels
        embed_dim = mdl_config.embed_dim
        dim_multiplier = tuple(mdl_config.dim_multiplier)
        num_conv_layers_per_stage = tuple(mdl_config.num_conv_layers)
        patch_size = mdl_config.stem.patch_size
        snn_cfg = mdl_config.snn

        num_stages = len(dim_multiplier)
        assert num_stages == 4
        assert len(num_conv_layers_per_stage) == num_stages

        self.stage_dims = [embed_dim * m for m in dim_multiplier]
        self.stages = nn.ModuleList()
        self._strides = []

        input_dim = in_channels
        stride = 1
        for stage_idx in range(num_stages):
            ds_factor = patch_size if stage_idx == 0 else 2
            stage_dim = self.stage_dims[stage_idx]

            self.stages.append(SNNCNNStage(
                dim_in=input_dim,
                dim_out=stage_dim,
                spatial_downsample_factor=ds_factor,
                num_conv_layers=num_conv_layers_per_stage[stage_idx],
                snn_cfg=snn_cfg,
            ))

            stride *= ds_factor
            self._strides.append(stride)
            input_dim = stage_dim

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

    def forward(self,
                x: th.Tensor,
                prev_states: Optional[SpikingStates] = None,
                token_mask: Optional[th.Tensor] = None,
                ) -> Tuple[BackboneFeatures, SpikingStates]:
        """
        Args:
            x: (N, C_in, H, W) event representation tensor
            prev_states: list of SpikingState per stage, or None
            token_mask: ignored (accepted for interface compatibility)
        Returns:
            features: {1: membrane_1, 2: membrane_2, 3: membrane_3, 4: membrane_4}
            states: list of SpikingState per stage
        """
        if prev_states is None:
            prev_states = [None] * self.num_stages

        output: Dict[int, FeatureMap] = {}
        states: SpikingStates = []

        for stage_idx, stage in enumerate(self.stages):
            membrane, spike, stage_mems = stage(x, prev_states[stage_idx])
            states.append(stage_mems)
            output[stage_idx + 1] = membrane  # membrane readout for FPN
            x = spike  # spike output to next stage

        return output, states
