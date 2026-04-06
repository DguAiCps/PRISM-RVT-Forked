# Adding a New Backbone Model

This guide covers the steps required to add a new recurrent backbone to the RVT detection framework.

## Files to modify/create

### 1. Backbone implementation
**`models/detection/recurrent_backbone/<name>.py`**

- Inherit from `BaseDetector` (`models/detection/recurrent_backbone/base.py`)
- Must implement:
  - `get_stage_dims(stages) -> Tuple[int, ...]`
  - `get_strides(stages) -> Tuple[int, ...]`
  - `forward(x, prev_states, token_mask) -> (BackboneFeatures, States)`
- `BackboneFeatures` = `Dict[int, FeatureMap]` (stage index 1-4 -> feature tensor)
- States = nested list of tensors, managed by RVT's `RNNStates`

**State management**: Do NOT detach states internally. The RVT framework handles truncated BPTT via `RNNStates.save_states_and_detach()` between sequences.

### 2. Register in `__init__.py`
**`models/detection/recurrent_backbone/__init__.py`**

```python
from .<module> import NewBackbone

def build_recurrent_backbone(backbone_cfg):
    ...
    elif name == 'NewBackbone':
        return NewBackbone(backbone_cfg)
```

### 3. Add to `config/modifier.py`
**`config/modifier.py`** — `dynamically_modify_train_config()`

Add the backbone name to the appropriate `elif` branch. All current SNN backbones need resolution aligned to stride 32:

```python
elif backbone_name in ('SNNCNN', 'SNNSwin', 'NewBackbone'):
    mdl_hw = _get_modified_hw_multiple_of(hw=dataset_hw, multiple_of=32)
    ...
```

### 4. Model config
**`config/model/<name>_yolox/default.yaml`**

```yaml
# @package _global_
defaults:
  - override /model: rnndet

model:
  backbone:
    name: NewBackbone
    input_channels: 20
    enable_masking: false
    # ... backbone-specific params
  fpn:
    name: PAFPN
    compile: { enable: false, args: { mode: reduce-overhead } }
    depth: 0.67
    in_stages: [2, 3, 4]
    depthwise: false
    act: "silu"
  head:
    name: YoloX
    compile: { enable: false, args: { mode: reduce-overhead } }
    depthwise: false
    act: "silu"
  postprocess:
    confidence_threshold: 0.1
    nms_threshold: 0.45
```

### 5. Experiment config
**`config/experiment/gen1/<name>_default.yaml`**

```yaml
# @package _global_
defaults:
  - /model/<name>_yolox: default

training:
  precision: 16
  max_steps: 400000
  learning_rate: 0.0001
  lr_scheduler:
    use: True
    total_steps: ${..max_steps}
    pct_start: 0.005
    div_factor: 20
    final_div_factor: 10000
validation:
  check_val_every_n_epoch: 1
dataset:
  train:
    sampling: 'mixed'
    mixed: { w_stream: 1, w_random: 1 }
  eval:
    sampling: 'stream'
  ev_repr_name: 'stacked_histogram_dt=50_nbins=10'
  sequence_length: 21
```

### 6. Callbacks (if needed)
**`callbacks/detection.py`**

If the backbone has attention or special visualization, add a guard:
```python
if not hasattr(stage, 'blocks'):
    return None  # skip for non-transformer backbones
```

## Training command template

```bash
env CUDA_VISIBLE_DEVICES=0 nohup python -u train.py \
    model=rnndet dataset=gen1 dataset.path=/path/to/gen1_preprocessed \
    wandb.project_name=RVT wandb.group_name=gen1_<name> \
    +experiment/gen1=<name>_default.yaml \
    hardware.gpus=[0] batch_size.train=4 batch_size.eval=4 \
    hardware.num_workers.train=6 hardware.num_workers.eval=2 \
    > train_<name>_log.txt 2>&1 &
```

## Visualization command template

```bash
python visualize.py dataset=gen1 +model/<name>_yolox=default \
    'checkpoint="path/to/ckpt.ckpt"' \
    'dataset.path=/path/to/gen1_preprocessed' \
    hardware.gpus=0 batch_size.eval=1 hardware.num_workers.eval=0 \
    +max_frames=1000 +save_video=true
```

## Checklist

- [ ] Backbone class inherits `BaseDetector`, implements required methods
- [ ] Registered in `__init__.py`
- [ ] Added to `modifier.py` backbone name check
- [ ] Model config YAML created
- [ ] Experiment config YAML created
- [ ] No internal state detach (let RVT framework handle it)
