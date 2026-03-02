# SNN Backbone Implementation Plan

Detailed, file-level plan for integrating an SNN-CNN backbone into the RVT codebase.
Companion to [snn_backbone_architecture.md](snn_backbone_architecture.md).

## Guiding Principle

**Minimal invasion.** Every change is additive or extends an existing branch.
No existing file is restructured. The MaxViTRNN path remains fully functional.

---

## 1. Integration Points (files touched)

| File | Change Type | Risk |
|------|-------------|------|
| `models/layers/spiking.py` | **NEW** | None (new file) |
| `models/detection/recurrent_backbone/snn_cnn.py` | **NEW** | None (new file) |
| `models/detection/recurrent_backbone/__init__.py` | Add `elif` branch | Low (2 lines) |
| `config/modifier.py` | Add `elif` branch | Low (5 lines) |
| `config/model/snn_yolox/default.yaml` | **NEW** | None (new file) |
| `config/experiment/gen1/snn_tiny.yaml` | **NEW** | None (new file) |
| `requirements.txt` or env | Add `snntorch` | Low |

**Files NOT touched:**
- `modules/detection.py` (training loop)
- `modules/utils/detection.py` (RNNStates, BackboneFeatureSelector)
- `data/` (all data loading)
- `models/detection/yolox_extension/` (FPN, head, detector)
- `train.py`
- `config/model/rnndet.yaml`
- `config/model/maxvit_yolox/default.yaml`

---

## 2. New Files

### 2.1 `models/layers/spiking.py` — LIF Neuron Layer

Thin wrapper around `snntorch.Leaky` that matches the DWSConvLSTM2d usage pattern:
takes `(x, prev_state)`, returns `(output, new_state)`.

```python
"""Spiking neuron layers for SNN backbone."""
import torch
import torch.nn as nn
import snntorch as snn


class SpikingConvBlock(nn.Module):
    """Conv2d + BatchNorm2d + LIF neuron.

    Input:  (N, C_in, H, W) float tensor  (event repr or spikes from prev layer)
    Output: spike (N, C_out, H', W') binary, membrane (N, C_out, H', W') float
    State:  membrane potential V_t of shape (N, C_out, H', W')
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 padding: int = 1,
                 beta_init: float = 0.9,
                 learn_beta: bool = True,
                 threshold: float = 1.0,
                 reset_mechanism: str = 'subtract'):  # 'subtract' or 'zero'
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.lif = snn.Leaky(
            beta=beta_init,
            learn_beta=learn_beta,
            threshold=threshold,
            reset_mechanism=reset_mechanism,
        )

    def forward(self, x: torch.Tensor, mem: torch.Tensor = None):
        """
        Args:
            x: input tensor (N, C_in, H, W)
            mem: previous membrane potential (N, C_out, H', W') or None
        Returns:
            spike: (N, C_out, H', W') binary
            mem: (N, C_out, H', W') float (new membrane potential)
        """
        cur = self.bn(self.conv(x))
        if mem is None:
            spike, mem = self.lif(cur)        # snntorch auto-inits membrane
        else:
            spike, mem = self.lif(cur, mem)
        return spike, mem
```

Key points:
- `snntorch.Leaky` handles surrogate gradient automatically during backprop
- `beta` (leak factor) is learnable per-layer via `learn_beta=True`
- `reset_mechanism='subtract'` subtracts `V_thresh` on spike (soft reset), preserving residual charge
- BatchNorm before LIF for threshold balancing
- State shape matches Conv2d output: `(N, C_out, H', W')`

### 2.2 `models/detection/recurrent_backbone/snn_cnn.py` — SNN-CNN Backbone

```python
"""SNN-CNN backbone: drop-in replacement for MaxViTRNN (RNNDetector)."""
from typing import Dict, List, Optional, Tuple

import torch as th
import torch.nn as nn
from omegaconf import DictConfig

from data.utils.types import FeatureMap, BackboneFeatures
from models.layers.spiking import SpikingConvBlock
from .base import BaseDetector

# Type aliases for SNN states
SpikingState = List[Optional[th.Tensor]]       # membrane potentials within one stage
SpikingStates = List[SpikingState]              # all stages


class SNNCNNStage(nn.Module):
    """One stage of the SNN-CNN backbone.

    Architecture per stage:
        Conv2d(stride) + BN + LIF   (downsample + first spiking layer)
        Conv2d(1)      + BN + LIF   (second spiking layer, same spatial dims)

    Input format:  NCHW (float — event tensor or spikes from previous stage)
    Output format: NCHW
    Feature output: membrane potential of LAST LIF neuron (continuous, for FPN)
    Spike output:   spike of LAST LIF neuron (binary, passed to next stage)
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
            # Stem: use 7x7 kernel with stride 4 (overlapping, like MaxViTRNN)
            k, p = 7, 3
        else:
            # Stages 1-3: use 3x3 kernel with stride 2
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
                token_mask=None,  # accepted for interface compat, ignored
                ) -> Tuple[FeatureMap, th.Tensor, SpikingState]:
        """
        Args:
            x: (N, C_in, H, W) input tensor
            prev_mems: list of membrane potentials, one per spiking layer, or None
        Returns:
            feature: (N, C_out, H', W') membrane potential of last layer (for FPN)
            spike:   (N, C_out, H', W') spike output of last layer (for next stage)
            new_mems: list of new membrane potentials
        """
        if prev_mems is None:
            prev_mems = [None] * self.num_conv_layers

        new_mems = []
        for i, layer in enumerate(self.layers):
            spike, mem = layer(x, prev_mems[i])
            new_mems.append(mem)
            x = spike  # spike output feeds into next layer

        # Last layer's membrane = continuous feature for FPN
        # Last layer's spike = input to next stage
        return mem, spike, new_mems


class SNNCNNBackbone(BaseDetector):
    """SNN-CNN backbone with 4 stages, matching RNNDetector interface.

    Config keys (under model.backbone):
        name: "SNNCNN"
        input_channels: 20
        embed_dim: 64
        dim_multiplier: [1, 2, 4, 8]
        num_conv_layers: [2, 2, 2, 2]   # conv+LIF layers per stage
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
        return tuple(self.stage_dims[i] for i in stage_indices)

    def get_strides(self, stages: Tuple[int, ...]) -> Tuple[int, ...]:
        stage_indices = [x - 1 for x in stages]
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
            features: {1: (N,C1,H/4,W/4), 2: ..., 3: ..., 4: ...}
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
```

Key design decisions:
- **`membrane` goes to FPN** (continuous, rich features, clean gradient path)
- **`spike` goes to next stage** (binary, sparse, SNN-native communication)
- **State is a nested list**: `SpikingStates = List[List[Optional[Tensor]]]`
  - Outer list: per stage (4 elements)
  - Inner list: per spiking layer within stage (e.g., 2 elements if `num_conv_layers=2`)
  - `RNNStates.recursive_detach/reset` handles this via duck typing on Lists/Tensors
- **Stage 0 uses 7x7 conv with stride 4** (same receptive field as MaxViTRNN stem)
- **`token_mask` is accepted but ignored** — masking is a MaxViT-specific feature

### 2.3 `config/model/snn_yolox/default.yaml`

```yaml
# @package _global_
model:
  name: rnndet
  backbone:
    name: SNNCNN
    input_channels: 20
    embed_dim: 64
    dim_multiplier: [1, 2, 4, 8]
    num_conv_layers: [2, 2, 2, 2]
    enable_masking: false
    stem:
      patch_size: 4
    snn:
      beta_init: 0.9
      learn_beta: true
      threshold: 1.0
      reset_mechanism: subtract
  fpn:
    name: PAFPN
    compile:
      enable: false
      args:
        mode: reduce-overhead
    depth: 0.67
    in_stages: [2, 3, 4]
    depthwise: false
    act: silu
  head:
    name: YoloX
    compile:
      enable: false
      args:
        mode: reduce-overhead
    depthwise: false
    act: silu
    num_classes: ???
  postprocess:
    confidence_threshold: 0.1
    nms_threshold: 0.45
```

Note: `model.name` stays `rnndet`. The SNN backbone is selected by `backbone.name: SNNCNN`.
This means the existing training loop in `modules/detection.py` is used as-is.

### 2.4 `config/experiment/gen1/snn_tiny.yaml`

```yaml
# @package _global_
defaults:
  - default

model:
  backbone:
    embed_dim: 32
  fpn:
    depth: 0.33
```

Overrides for tiny variant (embed_dim=32 like RVT-T for fair comparison).

---

## 3. Modifications to Existing Files

### 3.1 `models/detection/recurrent_backbone/__init__.py`

Current (line 6-11):
```python
def build_recurrent_backbone(backbone_cfg: DictConfig):
    name = backbone_cfg.name
    if name == 'MaxViTRNN':
        return MaxViTRNNDetector(backbone_cfg)
    else:
        raise NotImplementedError
```

Add SNN branch:
```python
from .maxvit_rnn import RNNDetector as MaxViTRNNDetector
from .snn_cnn import SNNCNNBackbone                          # NEW

def build_recurrent_backbone(backbone_cfg: DictConfig):
    name = backbone_cfg.name
    if name == 'MaxViTRNN':
        return MaxViTRNNDetector(backbone_cfg)
    elif name == 'SNNCNN':                                    # NEW
        return SNNCNNBackbone(backbone_cfg)                   # NEW
    else:
        raise NotImplementedError
```

### 3.2 `config/modifier.py`

Current (line 24-44):
```python
if mdl_name == 'rnndet':
    backbone_cfg = mdl_cfg.backbone
    backbone_name = backbone_cfg.name
    if backbone_name == 'MaxViTRNN':
        # ... partition size calculation ...
    else:
        print(f'{backbone_name=} not available')
        raise NotImplementedError
```

Add SNN branch:
```python
if mdl_name == 'rnndet':
    backbone_cfg = mdl_cfg.backbone
    backbone_name = backbone_cfg.name
    if backbone_name == 'MaxViTRNN':
        # ... existing partition size calculation (unchanged) ...
    elif backbone_name == 'SNNCNN':                                  # NEW
        # SNN-CNN does not need partition sizes or resolution alignment.
        # Just ensure in_res_hw is set for InputPadderFromShape.
        # Round up to multiple of 32 (total stride) for clean division.
        mdl_hw = _get_modified_hw_multiple_of(hw=dataset_hw, multiple_of=32)
        print(f'Set {backbone_name} backbone (height, width) to {mdl_hw}')
        backbone_cfg.in_res_hw = mdl_hw
    else:
        print(f'{backbone_name=} not available')
        raise NotImplementedError
    num_classes = 2 if dataset_name == 'gen1' else 3
    mdl_cfg.head.num_classes = num_classes
    print(f'Set {num_classes=} for detection head')
```

---

## 4. State Compatibility Verification

The `RNNStates` class in `modules/utils/detection.py` uses recursive duck typing:

```python
# recursive_detach handles: Tensor, List, Tuple, Dict
# recursive_reset handles: Tensor (sets to 0), List, Tuple, Dict
```

SNN state structure: `SpikingStates = List[List[Optional[Tensor]]]`

Trace through operations:
1. **`save_states_and_detach(worker_id, states)`**:
   - `states` is `List[List[Optional[Tensor]]]`
   - `recursive_detach(List)` → `[recursive_detach(x) for x in List]`
   - `recursive_detach(List[Optional[Tensor]])` → `[recursive_detach(x) for x in ...]`
   - `recursive_detach(Tensor)` → `Tensor.detach()`
   - `recursive_detach(None)` → **PROBLEM: None is not handled!**

**Fix needed:** When `prev_states` is `None` at the very first step, `get_states()` returns `None`
and the backbone handles it. But after the first forward pass, all membrane tensors are populated
(snntorch.Leaky auto-initializes membrane to zeros). So `None` only appears on the first call,
which is handled by `get_states()` returning `None` → backbone sees `None` → auto-initializes.

After the first forward: all states are `List[List[Tensor]]` (no `None` inside) → safe.

**Verification**: snntorch.Leaky always returns a membrane tensor (never None) when given input,
so after the first timestep, the inner lists contain only Tensors. Confirmed safe.

---

## 5. What About `LstmStates` Type Hints?

`detector.py` and `modules/detection.py` use `LstmStates` in type hints:
```python
def forward_backbone(self, x, previous_states: Optional[LstmStates] = None) -> Tuple[BackboneFeatures, LstmStates]:
```

These are **just type annotations** — Python does not enforce them at runtime.
The SNN states (nested lists of tensors) will flow through without issues.

No changes needed. We can optionally add a more general type alias later:
```python
RecurrentStates = Union[LstmStates, SpikingStates]
```
But this is cosmetic and not required for the initial experiment.

---

## 6. Implementation Order

### Step 1: Install snntorch
```bash
pip install snntorch
```

### Step 2: Create `models/layers/spiking.py`
- SpikingConvBlock class
- Unit test: verify forward/backward with random input, check spike is binary, check membrane is float

### Step 3: Create `models/detection/recurrent_backbone/snn_cnn.py`
- SNNCNNStage and SNNCNNBackbone classes
- Unit test: verify output shapes match expected BackboneFeatures format
- Unit test: verify state shapes and that states survive recursive_detach/reset

### Step 4: Register backbone in `__init__.py`
- Add 2-line elif branch

### Step 5: Create config files
- `config/model/snn_yolox/default.yaml`
- `config/experiment/gen1/snn_tiny.yaml`

### Step 6: Modify `config/modifier.py`
- Add elif branch for SNNCNN resolution handling

### Step 7: Integration test
- Run training for ~100 steps with SNN backbone on Gen1
- Verify: loss decreases, no NaN, GPU memory reasonable
- Monitor spike rates per stage (add temporary logging)

### Step 8: Full training run
- Train SNN-CNN-Tiny on Gen1 (same schedule as RVT-T: 400K steps)
- Compare mAP against RVT-T baseline

---

## 7. Training Command

```bash
env CUDA_VISIBLE_DEVICES=2,3 /rise/RISE3/.conda/envs/python3_12/bin/python -u train.py \
    model=rnndet \
    dataset=gen1 \
    dataset.path=/rise/RISE3/datasets/gen1_preprocessed \
    wandb.project_name=RVT \
    wandb.group_name=gen1_snn \
    +experiment/gen1="snn_tiny.yaml" \
    +model/snn_yolox="default.yaml" \
    hardware.gpus=[0,1] \
    batch_size.train=4 \
    batch_size.eval=4 \
    hardware.num_workers.train=6 \
    hardware.num_workers.eval=2 \
    training.learning_rate=1e-4
```

Note: `+model/snn_yolox=default.yaml` overrides the default backbone config with SNN config.
The exact Hydra override syntax may need adjustment based on the config group structure.

---

## 8. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| snntorch not compatible with PyTorch 2.5.1 | Low | High | Check compatibility before starting; fallback: implement LIF manually (~30 lines) |
| Surrogate gradient vanishing over 21 steps | Medium | Medium | Monitor grad norms; reduce sequence_length if needed |
| Spike rate too low (dead neurons) | Medium | Medium | BatchNorm before LIF; tune threshold; monitor rates |
| FPN unstable with membrane readout | Low | Medium | Membrane is continuous and bounded — similar to LSTM hidden state |
| State management breaks with nested lists | Low | High | Pre-verified: RNNStates handles nested lists via duck typing |
| Config override conflicts with Hydra | Medium | Low | Test config resolution before full training |

---

## 9. Validation Checklist

Before starting full training, verify all of these:

- [ ] `snntorch` installed and importable in python3_12 env
- [ ] `SpikingConvBlock`: forward produces binary spikes and float membrane
- [ ] `SpikingConvBlock`: backward computes gradients (surrogate gradient works)
- [ ] `SNNCNNBackbone`: output dict has keys {1, 2, 3, 4} with correct shapes
- [ ] `SNNCNNBackbone`: `get_stage_dims((2,3,4))` returns correct channel dims
- [ ] `SNNCNNBackbone`: `get_strides((2,3,4))` returns `(8, 16, 32)`
- [ ] `SNNCNNBackbone`: states survive `RNNStates.recursive_detach()`
- [ ] `SNNCNNBackbone`: states survive `RNNStates.recursive_reset()`
- [ ] Config resolves correctly: `python train.py --cfg job model=rnndet +model/snn_yolox=default ...`
- [ ] Training starts without error for 10 steps
- [ ] Loss is finite and decreasing
- [ ] GPU memory usage is reasonable (~6GB per GPU for tiny model)
