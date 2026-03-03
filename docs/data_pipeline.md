# Data Pipeline: Event Representation to Model

## Overview

This document traces the complete data flow from raw event camera data to the
model's forward pass, with tensor shapes at each stage.

## Event Representation: Stacked Histogram

**Config key:** `dataset.ev_repr_name = stacked_histogram_dt=50_nbins=10`

### Time Structure

```
1 sequence sample = 21 timesteps x 50ms = 1.05 seconds

Timestep:  t=0      t=1      t=2     ...                              t=20
         |--50ms--|--50ms--|--50ms--|- - - - - - - - - - - - - - - -|--50ms--|
```

- **dt=50** : each macro timestep covers a **50ms** window of events
- **sequence_length=21** : 21 consecutive timesteps per training sample

### Inside Each 50ms Timestep (10 Temporal Bins)

Each 50ms window is subdivided into **10 bins of 5ms each**, preserving
fine-grained temporal ordering within the timestep:

```
One timestep = 50ms
|--5ms--|--5ms--|--5ms--|--5ms--|--5ms--|--5ms--|--5ms--|--5ms--|--5ms--|--5ms--|
 bin 0   bin 1   bin 2   bin 3   bin 4   bin 5   bin 6   bin 7   bin 8   bin 9
```

For each bin, the representation counts **how many events** hit each pixel,
separately for each polarity (ON/OFF brightness change). Values are stored as
uint8 (0-255 event counts).

### Channel Layout (20 channels total)

```
Channel  0 = polarity 0 (OFF), bin 0  (0-5ms)
Channel  1 = polarity 0 (OFF), bin 1  (5-10ms)
  ...
Channel  9 = polarity 0 (OFF), bin 9  (45-50ms)
Channel 10 = polarity 1 (ON),  bin 0  (0-5ms)
Channel 11 = polarity 1 (ON),  bin 1  (5-10ms)
  ...
Channel 19 = polarity 1 (ON),  bin 9  (45-50ms)
```

**Why 10 bins instead of 1?**
A single histogram per 50ms would lose all temporal ordering within the
window -- the model couldn't distinguish "event at 2ms then nothing" from
"nothing then event at 48ms." The 10 bins preserve 5ms temporal resolution
while keeping the representation compact.

### Construction (data/utils/representations.py)

```python
# shape: (2 polarities, 10 bins, H, W) -> reshaped to (20, H, W)
representation = torch.zeros((2, 10, height, width), dtype=uint8)

# Normalize event timestamps to bin indices [0, 9]
t_norm = (time - t0) / (t1 - t0) * 10
t_idx  = floor(clamp(t_norm, max=9))

# Accumulate event counts per (polarity, bin, y, x)
representation.put_(pol * 10*H*W + t_idx * H*W + y * W + x, ones, accumulate=True)

# Merge to (20, H, W)
representation = representation.reshape(-1, H, W)
```

## Tensor Shape Flow

### Stage 1: H5 File -> Dataset

Preprocessed data is stored in H5 files. Each file contains the stacked
histogram representations for one sequence.

```
H5 file: (sequence_length, 20, 240, 304) uint8    [Gen1 resolution]
                            |    |    |
                            |    |    +-- width
                            |    +------- height
                            +------------ 2 polarities x 10 bins
```

**Source:** `data/genx_utils/sequence_base.py` -> `_get_event_repr_torch()`

### Stage 2: Dataloader Batch

The dataloader collates samples into batches:

```
batch['data'][DataType.EV_REPR]:
    List of length 21 (sequence_length), each element:
        tensor(batch_size, 20, 240, 304) uint8

batch['data'][DataType.OBJLABELS_SEQ]:
    SparselyBatchedObjectLabels for each timestep

batch['data'][DataType.IS_FIRST_SAMPLE]:
    Boolean tensor indicating sequence boundaries (for state reset)
```

### Stage 3: Training Step (modules/detection.py:131-159)

```python
for tidx in range(21):  # sequence_length
    ev_tensors = ev_tensor_sequence[tidx]          # (N, 20, 240, 304) uint8
    ev_tensors = ev_tensors.to(dtype=self.dtype)   # (N, 20, 240, 304) float16/32
    ev_tensors = self.input_padder.pad(ev_tensors) # (N, 20, 256, 320) padded

    features, states = backbone(ev_tensors, prev_states)
    prev_states = states  # membrane potentials carry over between timesteps
```

**Padding:** 240x304 -> 256x320 (add 16px to bottom and right) to match
`model.backbone.in_res_hw` which must be divisible by the total stride (4x2x2x2=32).

### Stage 4: SNNCNNBackbone Forward (per timestep)

```
Input:    (N, 20, 256, 320)
              |
Stage 0:  Conv2d(20->32, k=7, stride=4) + BN + LIF x2
          (N, 32, 64, 80)     -> output[1] = membrane for FPN
              | spike
Stage 1:  Conv2d(32->64, k=3, stride=2) + BN + LIF x2
          (N, 64, 32, 40)     -> output[2]
              | spike
Stage 2:  Conv2d(64->128, k=3, stride=2) + BN + LIF x2
          (N, 128, 16, 20)    -> output[3]
              | spike
Stage 3:  Conv2d(128->256, k=3, stride=2) + BN + LIF x2
          (N, 256, 8, 10)     -> output[4]
```

Total spatial downsampling: 4 x 2 x 2 x 2 = **32x**

### Stage 5: FPN + Detection Head

```
BackboneFeatures: {1: (N,32,64,80), 2: (N,64,32,40),
                   3: (N,128,16,20), 4: (N,256,8,10)}
    |
    v
PAFPN (uses stages 2,3,4)
    |
    v
YOLOX Head -> predictions + losses
```

Features from all labeled timesteps across the sequence are batched together
and processed by the detection head in a single forward pass.

## State Management Between Timesteps

- **Within a sequence (same batch):** membrane potentials carry over between
  timesteps. Gradients are truncated (detached) at timestep boundaries
  (truncated BPTT).
- **Between batches:** states are saved via `RNNStates.save_states_and_detach()`
  so membrane values persist across batch boundaries for streaming evaluation.
- **Sequence boundaries:** when `IS_FIRST_SAMPLE=True`, states are reset to
  zero via `RNNStates.reset()`.

## Gen1 Dataset Summary

| Parameter | Value |
|-----------|-------|
| Resolution | 240 x 304 |
| Padded resolution | 256 x 320 |
| Event representation | Stacked Histogram |
| dt (macro timestep) | 50ms |
| Temporal bins per timestep | 10 (5ms each) |
| Input channels | 20 (2 pol x 10 bins) |
| Sequence length | 21 timesteps (~1.05s) |
| Object classes | 2 (car, pedestrian) |
| Train sequences | 1457 |
| Val sequences | 428 |
