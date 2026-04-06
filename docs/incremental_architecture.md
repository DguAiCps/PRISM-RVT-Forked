# Incremental Architecture Search: SNN Backbones

## Motivation
The full SNN-Swin (SSA + windowed attention + spiking everything) was too radical
a jump from the feedforward SNN-CNN baseline. False positive detections on
background texture (trees, walls) suggest the model lacks temporal memory and/or
the attention mechanism is not properly normalized. Instead of debugging the
monolithic architecture, we build up from the validated CNN base one component at
a time.

## Steps

### Step 0: SNN-CNN (baseline, `SNNCNN`)
- Purely feedforward: `spike_t = LIF(Conv(x_t), mem_{t-1})`
- LIF membranes carry temporal state but no explicit recurrent feedback
- Already implemented in `snn_cnn.py`

### Step 1: SNN-CNN + Recurrent Feedback (`SNNCNNRecurrent`)
- Add per-stage feedback: output spikes are accumulated by a feedback LIF and
  re-injected into the next timestep's input
- `x_t' = x_t + fb_spike_{t-1}` where `fb_spike` comes from a LIF accumulating
  the stage's last-layer spike output
- **Validates**: Does recurrent feedback enable the model to track occluded
  objects (objects that had events at edges and then disappeared)?
- File: `snn_cnn.py` (extend `SNNCNNStage`)

### Step 2: Step 1 + Global SSA Attention
- Add a single global SSA attention block after CNN layers in each stage
- No windowing — attention sees full spatial extent
- **Validates**: Does cross-spatial attention improve over local convolutions?

### Step 3: Step 2 + Windowed Attention
- Replace global attention with shifted-window SSA
- Equivalent to current SNN-Swin but built on validated components
- **Validates**: Does windowed attention maintain accuracy while scaling?

## Design Constraints
- All models must implement `BaseDetector` interface (`get_stage_dims`,
  `get_strides`, `forward` returning `BackboneFeatures` and states)
- No internal per-timestep `detach()` — the RVT framework handles gradient
  truncation between batches via `RNNStates.save_states_and_detach()`. Within
  a training sequence (21 timesteps), gradients flow freely (standard BPTT).
- The redundant per-timestep detach in `snn_cnn.py` should NOT be replicated.
- States use flat lists compatible with `RNNStates.recursive_detach()`
- Config via Hydra YAML under `config/model/`
