"""SNN-CNN backbone: drop-in replacement for MaxViTRNN (RNNDetector).

Also contains SNNCNNRecurrentBackbone (Step 1 of incremental architecture search):
SNN-CNN + per-stage LIF feedback for temporal persistence.
"""
from typing import Dict, List, Optional, Tuple

import torch as th
import torch.nn as nn
from omegaconf import DictConfig

from data.utils.types import FeatureMap, BackboneFeatures
from models.layers.spiking import LIFNeuron, SpikingConvBlock, PlateauSpikingConvBlock
from models.layers.rnn import DWSConvLSTM2d
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
        channelwise_beta = snn_cfg.get('channelwise_beta', False)
        beta_spread = snn_cfg.get('beta_spread', 0.0)
        learn_reset = snn_cfg.get('learn_reset', False)
        reset_ratio_init = snn_cfg.get('reset_ratio_init', 1.0)
        reset_spread = snn_cfg.get('reset_spread', 0.0)

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
            channelwise_beta=channelwise_beta,
            beta_spread=beta_spread,
            learn_reset=learn_reset,
            reset_ratio_init=reset_ratio_init,
            reset_spread=reset_spread,
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
                channelwise_beta=channelwise_beta,
                beta_spread=beta_spread,
                learn_reset=learn_reset,
                reset_ratio_init=reset_ratio_init,
                reset_spread=reset_spread,
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
        # No internal detach — RVT framework handles truncated BPTT via
        # RNNStates.save_states_and_detach() between sequences.

        output: Dict[int, FeatureMap] = {}
        states: SpikingStates = []

        for stage_idx, stage in enumerate(self.stages):
            membrane, spike, stage_mems = stage(x, prev_states[stage_idx])
            states.append(stage_mems)
            output[stage_idx + 1] = membrane  # membrane readout for FPN
            x = spike  # spike output to next stage

        return output, states


# ---------------------------------------------------------------------------
# SNN-CNN + LSTM: SNN spatial features + ConvLSTM temporal memory
# ---------------------------------------------------------------------------

class SNNCNNLSTMBackbone(BaseDetector):
    """SNN-CNN backbone with per-stage ConvLSTM for explicit temporal memory.

    SNN-CNN stages extract spatial features (membrane + spike).
    ConvLSTM on each stage's membrane output provides long-term temporal
    memory that persists even when events stop.

    States per stage: [snn_mem_0, ..., snn_mem_{N-1}, lstm_h, lstm_c]
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
        self.lstm_cells = nn.ModuleList()
        self._strides = []
        self._num_snn_states = []

        input_dim = in_channels
        stride = 1
        for stage_idx in range(num_stages):
            ds_factor = patch_size if stage_idx == 0 else 2
            stage_dim = self.stage_dims[stage_idx]

            stage = SNNCNNStage(
                dim_in=input_dim,
                dim_out=stage_dim,
                spatial_downsample_factor=ds_factor,
                num_conv_layers=num_conv_layers_per_stage[stage_idx],
                snn_cfg=snn_cfg,
            )
            self.stages.append(stage)
            self._num_snn_states.append(stage.num_conv_layers)

            self.lstm_cells.append(DWSConvLSTM2d(dim=stage_dim))

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
        if prev_states is None:
            prev_states = [None] * self.num_stages

        output: Dict[int, FeatureMap] = {}
        states: SpikingStates = []

        for stage_idx, stage in enumerate(self.stages):
            n_snn = self._num_snn_states[stage_idx]

            # Unpack states: [snn_mems..., lstm_h, lstm_c]
            if prev_states[stage_idx] is not None:
                snn_mems = prev_states[stage_idx][:n_snn]
                lstm_state = (prev_states[stage_idx][n_snn],
                              prev_states[stage_idx][n_snn + 1])
            else:
                snn_mems = None
                lstm_state = None

            # SNN-CNN stage: spatial feature extraction
            membrane, spike, stage_mems = stage(x, snn_mems)

            # ConvLSTM: temporal memory on membrane features
            lstm_out, (new_h, new_c) = self.lstm_cells[stage_idx](membrane, lstm_state)

            # Pack states
            stage_mems.append(new_h)
            stage_mems.append(new_c)
            states.append(stage_mems)

            output[stage_idx + 1] = lstm_out  # LSTM output to FPN
            x = spike  # spikes feed next stage

        return output, states


# ---------------------------------------------------------------------------
# Step 1: SNN-CNN + Recurrent Feedback
# ---------------------------------------------------------------------------

class SNNCNNRecurrentStage(nn.Module):
    """SNN-CNN stage with direct spike feedback via channel concat.

    Same conv layers as SNNCNNStage, but the output spike is stored and
    concatenated with the next timestep's intermediate spike after the
    first (downsampling) layer. The second layer has in_channels = 2 * dim_out,
    giving it a learnable projection to fuse current and feedback spikes.

    No separate feedback LIF — the conv layer LIF membranes already provide
    multi-step temporal integration. The feedback spike is purely binary,
    1-step delayed, and the network learns how to weight it via the
    wider second conv layer.

    State list: [conv_mem_0, ..., conv_mem_{N-1}, prev_spike]
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
            k, p = 7, 3
        else:
            k, p = 3, 1

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

        # Second layer: accepts concat of spike (dim_out) + feedback (dim_out)
        if num_conv_layers > 1:
            layers.append(SpikingConvBlock(
                in_channels=dim_out * 2,
                out_channels=dim_out,
                kernel_size=3,
                stride=1,
                padding=1,
                beta_init=beta_init,
                learn_beta=learn_beta,
                threshold=threshold,
                reset_mechanism=reset_mechanism,
            ))

        # Remaining layers: standard dim_out -> dim_out
        for _ in range(num_conv_layers - 2):
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
        self.dim_out = dim_out

        # Total states: N conv membranes + prev_spike
        self.num_states = num_conv_layers + 1

    def forward(self,
                x: th.Tensor,
                prev_mems: Optional[SpikingState] = None,
                ) -> Tuple[FeatureMap, th.Tensor, SpikingState]:
        """
        Args:
            x: (N, C_in, H, W) input tensor (spike from previous stage)
            prev_mems: [conv_mem_0, ..., conv_mem_{N-1}, prev_spike] or None
        Returns:
            membrane: (N, C_out, H', W') for FPN
            spike: (N, C_out, H', W') for next stage
            new_mems: updated state list
        """
        if prev_mems is None:
            conv_mems = [None] * self.num_conv_layers
            prev_spike = None
        else:
            conv_mems = prev_mems[:self.num_conv_layers]
            prev_spike = prev_mems[self.num_conv_layers]

        # Feed output spike from previous timestep via channel concatenation.
        # prev_spike is in the output spatial dimensions (after downsample).
        # We concat AFTER the first layer (which does spatial downsampling),
        # so spatial dimensions match. The second layer accepts 2*dim_out channels.
        new_mems = []
        for i, layer in enumerate(self.layers):
            spike, mem = layer(x, conv_mems[i])
            new_mems.append(mem)
            x = spike

            # Concat feedback after the first (downsampling) layer
            if i == 0:
                if prev_spike is None:
                    prev_spike = th.zeros_like(x)
                x = th.cat([x, prev_spike], dim=1)  # (N, 2*C, H', W')

        # Store current output spike as feedback for next timestep
        new_mems.append(spike)

        return mem.clone(), spike, new_mems


class SNNCNNRecurrentBackbone(BaseDetector):
    """SNN-CNN with per-stage direct spike feedback (Step 1).

    Same architecture as SNNCNN but each stage stores its output spike
    and concatenates it into the next timestep's processing. The second
    conv layer in each stage has doubled input channels to accept both
    current and feedback spikes, learning a projection to fuse them.

    Temporal memory comes from two sources:
    1. Conv LIF membranes (multi-step, learnable beta decay)
    2. Direct spike feedback (1-step, binary)

    No internal per-timestep detach — relies on RVT framework for BPTT.

    Config keys (under model.backbone):
        name: "SNNCNNRecurrent"
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
            reset_mechanism: subtract
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

            self.stages.append(SNNCNNRecurrentStage(
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
            token_mask: ignored (interface compatibility)
        Returns:
            features: {1: membrane_1, ..., 4: membrane_4}
            states: list of SpikingState per stage
        """
        if prev_states is None:
            prev_states = [None] * self.num_stages
        # No internal detach — RVT framework handles BPTT via
        # RNNStates.save_states_and_detach() between batches.

        output: Dict[int, FeatureMap] = {}
        states: SpikingStates = []

        for stage_idx, stage in enumerate(self.stages):
            membrane, spike, stage_mems = stage(x, prev_states[stage_idx])
            states.append(stage_mems)
            output[stage_idx + 1] = membrane
            x = spike

        return output, states


# ---------------------------------------------------------------------------
# PlateauSNNCNN: SNN-CNN with PlateauIF + forget gate on last layer
# ---------------------------------------------------------------------------

class PlateauSNNCNNStage(nn.Module):
    """SNN-CNN stage with PlateauIF on the last conv layer.

    First N-1 layers use standard SpikingConvBlock (LIF with leak+reset).
    The last layer uses PlateauSpikingConvBlock (no leak, no reset,
    subtractive forget gate for saturation control).

    State list: [mem_0, ..., mem_{N-2}, v_pif, v_gate]
    Total states = (num_conv_layers - 1) + 2 = num_conv_layers + 1
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

        # Plateau-specific config
        plateau_cfg = snn_cfg.get('plateau', {})
        plateau_threshold = plateau_cfg.get('threshold', 1.0)
        gate_decay_init = plateau_cfg.get('gate_decay_init', 0.5)
        gate_threshold = plateau_cfg.get('gate_threshold', 0.5)
        tonic_init = plateau_cfg.get('tonic_init', 0.1)

        # First layer: spatial downsampling (always standard LIF)
        if spatial_downsample_factor == 4:
            k, p = 7, 3
        else:
            k, p = 3, 1

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

        # Middle layers: standard LIF
        for _ in range(num_conv_layers - 2):
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

        # Last layer: PlateauIF with forget gate
        self.plateau_layer = PlateauSpikingConvBlock(
            in_channels=dim_out,
            out_channels=dim_out,
            kernel_size=3,
            stride=1,
            padding=1,
            threshold=plateau_threshold,
            gate_decay_init=gate_decay_init,
            gate_threshold=gate_threshold,
            tonic_init=tonic_init,
        )

        # States: (N-1) LIF membranes + v_pif + v_gate
        self.num_states = num_conv_layers + 1

    def forward(self,
                x: th.Tensor,
                prev_mems: Optional[SpikingState] = None,
                ) -> Tuple[FeatureMap, th.Tensor, SpikingState]:
        if prev_mems is None:
            lif_mems = [None] * (self.num_conv_layers - 1)
            plateau_state = None
        else:
            lif_mems = prev_mems[:self.num_conv_layers - 1]
            plateau_state = (prev_mems[self.num_conv_layers - 1],
                             prev_mems[self.num_conv_layers])

        new_mems = []

        # Standard LIF layers
        for i, layer in enumerate(self.layers):
            spike, mem = layer(x, lif_mems[i])
            new_mems.append(mem)
            x = spike

        # Last layer: PlateauIF
        spike, v_pif, v_gate = self.plateau_layer(x, plateau_state)
        new_mems.append(v_pif)
        new_mems.append(v_gate)

        return v_pif.clone(), spike, new_mems


class PlateauSNNCNNBackbone(BaseDetector):
    """SNN-CNN backbone with PlateauIF + forget gate on the last layer of each stage.

    Standard LIF layers handle spatial feature extraction with leak+reset.
    The final layer per stage uses PlateauIF (no leak, no reset) for long-term
    memory preservation, with a subtractive forget gate to prevent saturation.

    Config keys (under model.backbone):
        name: "PlateauSNNCNN"
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
            reset_mechanism: subtract
            plateau:
                threshold: 1.0
                gate_decay_init: 0.5
                gate_threshold: 0.5
                tonic_init: 0.1
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

            self.stages.append(PlateauSNNCNNStage(
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
        if prev_states is None:
            prev_states = [None] * self.num_stages
        # No internal detach — RVT framework handles truncated BPTT via
        # RNNStates.save_states_and_detach() between sequences.
        # PlateauIF's forget gate needs temporal gradients to learn
        # proper forgetting dynamics across timesteps.

        output: Dict[int, FeatureMap] = {}
        states: SpikingStates = []

        for stage_idx, stage in enumerate(self.stages):
            membrane, spike, stage_mems = stage(x, prev_states[stage_idx])
            states.append(stage_mems)
            output[stage_idx + 1] = membrane
            x = spike

        return output, states
