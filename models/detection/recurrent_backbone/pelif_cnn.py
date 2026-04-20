"""PeLIF-CNN backbone: stacked PeLIF recurrent conv layers.

Architecture per stage:
    PeLIFConvBlock (downsample) → PeLIFConvBlock (refine) × (N-1)

Every layer has the same structure as sMNIST's PeLIFRecurrentLayer:
    u[t] = beta * u[t-1] + W_x(x[t]) + W_rec(s[t-1])
    s[t] = fire(u[t])  if t % P == 0 else 0

but in 2D convolutional form:
    W_x   = Conv2d (spatial feature extraction)
    W_rec = Conv2d (recurrence, kernel_size configurable: 1x1 or 3x3)

With rec_kernel_size=3, signals propagate spatially through recurrence.
Each layer adds 1 pixel of spatial reach per timestep.
N layers × T timesteps → effective RF expansion of ~2NT pixels.
"""
from typing import Dict, List, Optional, Tuple

import torch as th
import torch.nn as nn
from torch.nn.utils import spectral_norm
from omegaconf import DictConfig

from data.utils.types import FeatureMap, BackboneFeatures
from models.layers.pelif_spiking import PeLIFNeuron2d
from .base import BaseDetector

# State per layer: (membrane, prev_spike)
LayerState = Optional[Tuple[th.Tensor, th.Tensor]]
# State per stage: list of layer states
StageState = List[LayerState]
# State for whole backbone
PeLIFStates = List[StageState]


class PeLIFConvBlock(nn.Module):
    """Conv2d + PeLIF neuron + recurrent conv.

    Direct 2D conv analogue of PeLIFRecurrentLayer (src/layers/pelif_recurrent.py):
        I[t] = W_x(x[t]) + W_rec(s[t-1])
        u[t] = beta * u[t-1] + I[t]
        s[t] = fire(u[t] - v_th)  if t % P == 0 else 0
        u[t] -= s[t] * v_th

    Args:
        in_channels: input channels
        out_channels: output channels (= hidden size)
        kernel_size: spatial kernel for W_x
        stride: spatial stride for W_x (for downsampling)
        periods: clock periods for PeLIF neuron
        rec_kernel_size: kernel for W_rec (1=temporal only, 3=spatial propagation)
        beta_init: membrane decay init
        learn_beta: learnable decay per channel
        threshold: spike threshold
        surrogate: surrogate gradient type
        norm: 'bn' or 'none' — normalization on W_x output
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 periods: Tuple[int, ...] = (1, 2, 4, 8),
                 rec_kernel_size: int = 1,
                 beta_init: float = 0.9,
                 learn_beta: bool = True,
                 threshold: float = 1.0,
                 surrogate: str = 'triangle',
                 norm: str = 'bn'):
        super().__init__()

        padding = kernel_size // 2

        # W_x: feedforward spatial conv
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              bias=(norm == 'none'))
        if norm == 'none':
            self.bn = nn.Identity()
        else:
            self.bn = nn.BatchNorm2d(out_channels)

        # W_rec: recurrent conv with spectral norm
        rec_padding = rec_kernel_size // 2
        conv_rec = nn.Conv2d(out_channels, out_channels,
                             kernel_size=rec_kernel_size,
                             padding=rec_padding,
                             bias=False)
        nn.init.orthogonal_(conv_rec.weight.view(out_channels, -1))
        conv_rec.weight.data *= 0.5
        self.conv_rec = spectral_norm(conv_rec)

        # PeLIF neuron
        self.pelif = PeLIFNeuron2d(
            channels=out_channels,
            periods=periods,
            beta_init=beta_init,
            learn_beta=learn_beta,
            threshold=threshold,
            surrogate=surrogate,
        )

    def forward(self,
                x: th.Tensor,
                state: LayerState = None,
                t: int = 0,
                ) -> Tuple[th.Tensor, th.Tensor, LayerState]:
        """
        Args:
            x: (N, C_in, H, W) input
            state: (mem, prev_spike) or None
            t: current timestep
        Returns:
            spike: (N, C_out, H', W') binary
            mem: (N, C_out, H', W') continuous
            new_state: (mem, spike)
        """
        if state is not None:
            mem, s_prev = state
        else:
            mem, s_prev = None, None

        # I[t] = W_x(x[t]) + W_rec(s[t-1])
        cur = self.bn(self.conv(x))
        if s_prev is not None:
            cur = cur + self.conv_rec(s_prev)

        # u[t] = beta * u[t-1] + I[t], fire, reset
        spike, mem = self.pelif(cur, mem, t)

        return spike, mem, (mem, spike)


class PeLIFCNNStage(nn.Module):
    """Stage of stacked PeLIFConvBlocks.

    Architecture:
        PeLIFConvBlock (downsample, kernel=7 or 3) → PeLIFConvBlock (refine) × (N-1)

    Every layer is recurrent. With N layers and rec_kernel_size=3,
    each timestep propagates signals through N spatial hops.
    """

    def __init__(self,
                 dim_in: int,
                 dim_out: int,
                 spatial_downsample_factor: int,
                 num_layers: int,
                 periods: Tuple[int, ...],
                 pelif_cfg: DictConfig,
                 rec_kernel_size: int = 1,
                 norm: str = 'bn'):
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

        # First layer: downsample
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

        # Remaining layers: refine (stride=1)
        for _ in range(num_layers - 1):
            layers.append(PeLIFConvBlock(
                in_channels=dim_out,
                out_channels=dim_out,
                kernel_size=3,
                stride=1,
                **pelif_kwargs,
            ))

        self.layers = nn.ModuleList(layers)

    def forward(self,
                x: th.Tensor,
                prev_state: Optional[StageState] = None,
                t: int = 0,
                ) -> Tuple[FeatureMap, th.Tensor, StageState]:
        """
        Returns:
            membrane: continuous feature for FPN (from last layer)
            spike: binary for next stage input (from last layer)
            new_state: list of (mem, spike) per layer
        """
        if prev_state is None:
            prev_state = [None] * self.num_layers

        new_states = []
        for i, layer in enumerate(self.layers):
            spike, mem, layer_state = layer(x, prev_state[i], t)
            new_states.append(layer_state)
            x = spike  # spike feeds next layer

        # Last layer's membrane = stage output for FPN
        return mem, spike, new_states


class PeLIFCNNBackbone(BaseDetector):
    """PeLIF-CNN backbone: stacked PeLIF recurrent conv layers, 4 stages.

    Config keys (under model.backbone):
        name: "PeLIFCNN"
        input_channels: 20
        embed_dim: 64
        dim_multiplier: [1, 2, 4, 8]
        num_conv_layers: [2, 2, 2, 2]
        periods: [1, 2, 4, 8]
        rec_kernel_size: 1 or 3
        stem:
            patch_size: 4
        pelif:
            beta_init: 0.9
            learn_beta: true
            threshold: 1.0
            surrogate: triangle
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
        norm = mdl_config.get('norm', 'bn')

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
                num_layers=num_conv_layers_per_stage[stage_idx],
                periods=periods,
                pelif_cfg=pelif_cfg,
                rec_kernel_size=rec_kernel_size,
                norm=norm,
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
