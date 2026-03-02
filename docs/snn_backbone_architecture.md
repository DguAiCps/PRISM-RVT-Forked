# SNN Backbone for RVT: Architecture Design

## 1. Motivation

RVT (Recurrent Vision Transformers) achieves state-of-the-art event-based object detection using an LSTM-based recurrent backbone (MaxViTRNN) with a YOLOX detection head. The recurrent backbone processes event representations sequentially, maintaining temporal state across frames.

Spiking Neural Networks (SNNs) are a natural fit for event camera data:
- Event cameras produce sparse, asynchronous signals — similar to biological spikes
- SNNs process information through discrete spikes and membrane dynamics
- SNN backbones can be deployed on neuromorphic hardware (e.g., Intel Loihi, SynSense Speck) for low-power inference
- The RVT training framework (truncated BPTT, sparse supervision, streaming states) transfers directly to SNNs

This document outlines the architecture for replacing the LSTM-based MaxViTRNN backbone with an SNN backbone while retaining the existing FPN (PAFPN) and detection head (YOLOX).

## 2. Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Neuron model | LIF (Leaky Integrate-and-Fire) | Well-supported in snntorch, simple dynamics, proven in vision tasks |
| Training method | Surrogate gradient (BPTT) | Standard for SNN training, compatible with PyTorch autograd |
| Backbone-to-FPN interface | Membrane potential readout | Continuous-valued features per stage, directly compatible with ANN FPN |
| Internal communication | Spike-based between layers | Sparse activations, hardware-deployable |
| Initial experiment | Simple SNN-CNN backbone | Validate integration before adding attention mechanisms |

## 3. Architecture Overview

```
                         RVT Pipeline (unchanged)
                    ┌──────────────────────────────┐
                    │                              │
Event Tensor ──►[SNN Backbone]──► BackboneFeatures ──►[PAFPN]──►[YOLOX Head]──► Detections
  (N,20,H,W)     (NEW)         {2: feat, 3: feat,   (existing)  (existing)
                                 4: feat}
                    │                              │
                    └──────────────────────────────┘
                         States persist across
                         batches (streaming)
```

### 3.1 Hybrid SNN-ANN Design

```
┌─────────────────────────────────────────────┐
│              SNN Backbone                    │
│                                             │
│  Stage 0: SNN Conv Block (stride 4)         │
│     └─ SpikingConv2d → LIF → spike output   │
│     └─ Membrane readout ──────────────────── │──► feature_1 (not used by FPN)
│                                             │
│  Stage 1: SNN Conv Block (stride 2)         │
│     └─ SpikingConv2d → LIF → spike output   │
│     └─ Membrane readout ──────────────────── │──► feature_2 ──► PAFPN
│                                             │
│  Stage 2: SNN Conv Block (stride 2)         │
│     └─ SpikingConv2d → LIF → spike output   │
│     └─ Membrane readout ──────────────────── │──► feature_3 ──► PAFPN
│                                             │
│  Stage 3: SNN Conv Block (stride 2)         │
│     └─ SpikingConv2d → LIF → spike output   │
│     └─ Membrane readout ──────────────────── │──► feature_4 ──► PAFPN
│                                             │
└─────────────────────────────────────────────┘
```

### 3.2 Key Difference from MaxViTRNN

| Aspect | MaxViTRNN (current) | SNN Backbone (proposed) |
|--------|-------------------|----------------------|
| Temporal state | LSTM cell: (h_t, c_t) per stage | LIF membrane: V_t per layer |
| State shape | (N, C, H, W) per stage | (N, C, H, W) per spiking layer |
| Activation | Continuous (tanh, sigmoid gates) | Binary spikes + continuous membrane |
| Feature output | LSTM hidden state h_t | Membrane potential V_t |
| Attention | MaxViT window/grid attention | None in initial experiment |
| Inter-layer signal | Continuous feature maps | Spike tensors (0/1) |

## 4. LIF Neuron Model

### 4.1 Dynamics

The Leaky Integrate-and-Fire neuron follows:

```
V_t = beta * V_{t-1} + W * S_{t-1}^{input} - V_thresh * S_{t-1}^{output}
S_t = Theta(V_t - V_thresh)
```

Where:
- `V_t`: membrane potential at time t
- `beta`: leak factor (decay constant, 0 < beta < 1)
- `W * S_{t-1}^{input}`: weighted input current (from previous layer spikes or event tensor)
- `V_thresh`: firing threshold (typically 1.0)
- `S_t`: output spike (binary, 0 or 1)
- `Theta`: Heaviside step function (non-differentiable, replaced by surrogate during training)

### 4.2 Surrogate Gradient

During backpropagation, the Heaviside function is replaced with a smooth surrogate:

```
d_Theta/d_V ≈ (1 / (pi * (1 + (pi * slope * (V - V_thresh))^2)))   [ATan surrogate]
```

snntorch provides multiple surrogate options (ATan, FastSigmoid, Triangular). ATan is a reasonable default.

### 4.3 Membrane Potential Readout

At each stage output, instead of passing spikes to the FPN, we read the membrane potential V_t directly:

```
spike, V_t = LIF(input_current, V_{t-1})
feature_output = V_t    # continuous-valued, compatible with ANN FPN
spike_output = spike     # passed to next SNN stage as input
```

This preserves:
- Rich spatial information (membrane encodes accumulated evidence)
- Gradient flow through the readout (no surrogate needed at the output)
- Compatibility with the existing PAFPN which expects continuous feature maps

## 5. Interface Contract

### 5.1 Required Methods

The SNN backbone must implement the same interface as RNNDetector:

```python
class SNNBackbone(nn.Module):
    """
    Drop-in replacement for RNNDetector (MaxViTRNN).
    """

    def forward(
        self,
        x: th.Tensor,                              # (N, C_in, H, W) event tensor
        prev_states: Optional[List] = None,         # list of membrane potentials
        token_mask: Optional[th.Tensor] = None      # (N, H', W') binary mask
    ) -> Tuple[BackboneFeatures, List]:
        """
        Returns:
            features: Dict[int, Tensor]  -- {1: (N,C1,H/4,W/4), 2: ..., 3: ..., 4: ...}
            states: List                 -- membrane potentials for all spiking layers
        """
        ...

    def get_stage_dims(self, stages: Tuple[int, ...]) -> Tuple[int, ...]:
        """Return channel dimensions for requested 1-based stage indices."""
        ...

    def get_strides(self, stages: Tuple[int, ...]) -> Tuple[int, ...]:
        """Return cumulative spatial strides for requested stages."""
        ...
```

### 5.2 State Type

```python
# Current RVT type:
LstmState = Optional[Tuple[Tensor, Tensor]]    # (h, c)
LstmStates = List[LstmState]                   # per stage

# SNN equivalent:
SpikingState = Optional[Tensor]                 # V_membrane per layer
SpikingStates = List[SpikingState]              # per spiking layer (may be >4 if multiple layers per stage)
```

The existing `RNNStates` utility class uses `recursive_detach()` and `recursive_reset()` which operate on Tensors/Lists/Tuples/Dicts via duck typing. SNN membrane states (plain Tensors or Lists of Tensors) are fully compatible without modification.

### 5.3 Output Feature Dimensions

Must match the FPN expectation:

| Stage | Spatial Size (Gen1: 240x304) | Required Channels | Stride |
|-------|------------------------------|-------------------|--------|
| 1 | H/4 x W/4 = 60 x 76 | embed_dim * 1 | 4 |
| 2 | H/8 x W/8 = 30 x 38 | embed_dim * 2 | 8 |
| 3 | H/16 x W/16 = 15 x 19 | embed_dim * 4 | 16 |
| 4 | H/32 x W/32 = 8 x 10 | embed_dim * 8 | 32 |

FPN uses stages [2, 3, 4]. Stage 1 features are computed but not consumed by the FPN.

## 6. Initial Experiment: SNN-CNN Backbone

Before implementing attention mechanisms, start with a simple convolutional SNN backbone to validate the integration.

### 6.1 Stage Design (Simple SNN-CNN)

Each stage consists of:

```
Input (spike tensor or event tensor for stage 0)
  │
  ├─ Conv2d (stride=4 for stage 0, stride=2 for stages 1-3)
  ├─ BatchNorm2d
  ├─ LIF neuron → spike + membrane
  │
  ├─ Conv2d (stride=1, same spatial)
  ├─ BatchNorm2d
  ├─ LIF neuron → spike + membrane
  │
  └─ Output:
       spike → next stage input
       membrane (V_t) → BackboneFeatures[stage_idx]
```

### 6.2 Design Choices for Initial Experiment

- **No attention blocks**: Pure convolutional, validating SNN-FPN integration first
- **2 conv layers per stage**: Simple but sufficient to test gradient flow
- **Standard LIF**: Default snntorch.Leaky with ATan surrogate
- **Learnable beta**: Allow the leak factor to be learned per layer
- **BatchNorm before LIF**: Stabilizes membrane dynamics (threshold-balancing)
- **No residual connections initially**: Add later if gradient issues arise

### 6.3 Expected Parameter Count

With embed_dim=64, dim_multiplier=[1,2,4,8]:

| Stage | Conv Params (approx) | Notes |
|-------|---------------------|-------|
| 0 | 20 * 64 * 7 * 7 + 64 * 64 * 3 * 3 = ~100K | 7x7 stem + 3x3 conv |
| 1 | 64 * 128 * 3 * 3 + 128 * 128 * 3 * 3 = ~220K | Two 3x3 convs |
| 2 | 128 * 256 * 3 * 3 + 256 * 256 * 3 * 3 = ~885K | Two 3x3 convs |
| 3 | 256 * 512 * 3 * 3 + 512 * 512 * 3 * 3 = ~3.5M | Two 3x3 convs |
| **Total** | **~4.7M** | Comparable to MaxViTRNN tiny (4.4M) |

LIF neurons add negligible parameters (only beta per layer if learnable).

## 7. Training Considerations

### 7.1 What Stays the Same

- **Training loop**: `modules/detection.py` training_step unchanged
- **State management**: `RNNStates` class works with membrane potentials as-is
- **Sparse supervision**: Loss computed only at frames with GT labels
- **Truncated BPTT**: States detached at sequence boundaries (every 21 frames)
- **Streaming datapipe**: Worker-based state persistence unchanged
- **Detection head**: PAFPN + YOLOX head + NMS post-processing unchanged
- **Data augmentation**: Event representation augmentation unchanged

### 7.2 SNN-Specific Training Concerns

**Surrogate gradient stability**:
- LIF surrogate gradients can vanish over long sequences
- Mitigation: sequence_length=21 is relatively short (bounded BPTT)
- Monitor gradient norms per stage during early training
- If needed, reduce sequence_length or use PLIF (learnable time constants)

**Threshold balancing**:
- BatchNorm before LIF helps keep inputs in a range where neurons fire at reasonable rates
- Monitor spike rates: too low (<5%) means information loss, too high (>95%) means no sparsity benefit
- Target: 20-50% firing rate per layer

**Learning rate**:
- SNN training typically benefits from slightly lower learning rates than ANN
- Start with 1e-4 (RVT default is 2e-4) and tune

**Beta initialization**:
- Initialize leak factor beta ~0.9 (slow leak, longer memory)
- Higher beta = longer temporal integration, useful for event data
- Allow per-layer learning of beta

### 7.3 Membrane Potential Readout Gradient Flow

```
                     surrogate gradient
LIF forward:  V_t ──────► Theta(V_t - V_thresh) ──► spike ──► next stage
                │
                └──────────────────────────────────► V_t (membrane readout) ──► FPN
                           direct gradient path
```

The membrane readout path has clean gradients (no surrogate needed), which helps the FPN and detection head train stably. The surrogate gradient path trains the SNN's spiking dynamics.

## 8. Neuromorphic Deployment Path

The hybrid SNN-ANN design enables a staged deployment strategy:

### Phase 1: GPU Training (current focus)
- Train full pipeline on GPU with surrogate gradients
- SNN backbone + ANN FPN/Head
- Validate detection accuracy on Gen1 dataset

### Phase 2: Backbone Export to Neuromorphic Hardware
- Export trained SNN backbone weights
- Deploy backbone on neuromorphic chip (e.g., Loihi, Speck)
- FPN + Head remain on conventional hardware (or edge GPU)

### Phase 3: Full Event-Driven Pipeline (future)
- Replace frame-based event representations with raw event streams
- SNN backbone processes events asynchronously
- Trigger detection head at fixed intervals or adaptively

### Hardware Compatibility Notes
- LIF neurons map directly to neuromorphic primitives
- Conv layers are supported on most neuromorphic platforms
- Membrane readout may need platform-specific implementation
- Attention mechanisms (if added later) may not have direct hardware mapping

## 9. File Structure (Planned)

```
models/
  detection/
    recurrent_backbone/
      maxvit_rnn.py              # existing LSTM backbone
      snn_backbone.py            # NEW: SNN-CNN backbone
    build_backbone.py            # modify to support SNN backbone selection
  layers/
    rnn.py                       # existing LSTM cell
    spiking.py                   # NEW: LIF neuron wrappers (snntorch-based)

config/
  model/
    snn_yolox/
      default.yaml               # NEW: SNN backbone config
    rnndet.yaml                  # existing (unchanged)

modules/
  detection.py                   # unchanged (state management is backbone-agnostic)
  utils/
    detection.py                 # unchanged (RNNStates works with any tensor states)
```

## 10. Evaluation Plan

### Metrics
- **mAP** on Gen1 test set (primary metric, compare against RVT-T baseline)
- **Spike rate** per stage (monitor sparsity)
- **Gradient norm** per stage (monitor training stability)
- **Inference latency** on GPU (baseline for hardware comparison)
- **Parameter count** and **FLOPs/SOPs** (spike operations)

### Baselines
- RVT-T (MaxViTRNN tiny, embed_dim=32): published mAP on Gen1
- Current training run: will provide local baseline

### Ablation Studies (future)
- Impact of beta (leak factor) on temporal integration
- Membrane readout vs spike rate coding
- Sequence length sensitivity (7, 14, 21, 42)
- Number of conv layers per stage (1, 2, 3)

## 11. References

- RVT: Gehrig & Scaramuzza, "Recurrent Vision Transformers for Object Detection with Event Cameras", CVPR 2023
- snntorch: Eshraghian et al., "Training Spiking Neural Networks Using Lessons From Deep Learning", Proc. IEEE 2023
- Spiking-YOLO: Kim et al., "Spiking-YOLO: Spiking Neural Network for Energy-Efficient Object Detection", AAAI 2020
- EMS-YOLO: Su et al., "Deep Directly-Trained Spiking Neural Networks for Object Detection", ICCV 2023
- Spikformer: Zhou et al., "Spikformer: When Spiking Neural Network Meets Transformer", ICLR 2023
