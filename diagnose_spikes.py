"""Spike diagnostic script for SNNCNNRecurrent backbone.

Captures output spikes and feedback (prev_spike) across timesteps to verify
that the direct spike feedback mechanism carries meaningful temporal
information and doesn't degenerate into noise or oscillation.

Outputs:
    1. Spatial heatmaps of output and feedback spikes (per stage, per timestep)
    2. Spike raster plots (channel activity vs timestep)
    3. Firing rate plots over time
    4. Summary statistics printed to console

Usage:
    python diagnose_spikes.py dataset=gen1 +model/snn_recurrent_yolox=default \
        'checkpoint="path/to/ckpt.ckpt"' \
        dataset.path=/path/to/gen1_preprocessed \
        hardware.gpus=0 batch_size.eval=1 hardware.num_workers.eval=0 \
        +max_batches=5
"""
import os

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['OMP_NUM_THREADS'] = '1'

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.backends import cuda, cudnn
from einops import rearrange, reduce

cuda.matmul.allow_tf32 = True
cudnn.allow_tf32 = True
torch.multiprocessing.set_sharing_strategy('file_system')

import hydra
from omegaconf import DictConfig, OmegaConf, open_dict

from config.modifier import dynamically_modify_train_config
from data.utils.types import DataType, DatasetSamplingMode
from modules.utils.fetch import fetch_data_module
from modules.detection import Module as DetectionModule
from utils.padding import InputPadderFromShape
from modules.utils.detection import RNNStates


# ---------------------------------------------------------------------------
# Hook to capture internal spike data from SNNCNNRecurrentStage
# ---------------------------------------------------------------------------

class SpikeCaptureHook:
    """Monkey-patches SNNCNNRecurrentStage.forward to capture spike data.

    New state layout: [conv_mem_0, ..., conv_mem_{N-1}, prev_spike]
    prev_spike is the direct output spike stored for feedback (no fb_lif).
    """

    def __init__(self):
        self.timesteps: List[Dict] = []
        self._current: Dict = {}

    def new_timestep(self):
        if self._current:
            self.timesteps.append(self._current)
        self._current = {}

    def finalize(self):
        if self._current:
            self.timesteps.append(self._current)
            self._current = {}

    def record(self, stage_idx: int, membrane, spike, prev_spike):
        self._current[stage_idx] = {
            'output_spike': spike.detach().cpu(),
            'prev_spike': prev_spike.detach().cpu() if prev_spike is not None else None,
            'membrane': membrane.detach().cpu(),
        }


def patch_backbone_for_capture(backbone, capture: SpikeCaptureHook):
    """Monkey-patch each stage's forward to capture spike data."""
    for stage_idx, stage in enumerate(backbone.stages):
        original_forward = stage.forward

        def make_wrapper(orig_fwd, si):
            def wrapped_forward(x, prev_mems=None):
                # Get prev_spike before forward (last element of state)
                prev_spike_before = None
                if prev_mems is not None:
                    prev_spike_before = prev_mems[-1]  # last element is prev_spike

                membrane, spike, new_mems = orig_fwd(x, prev_mems)
                capture.record(si, membrane, spike, prev_spike_before)
                return membrane, spike, new_mems
            return wrapped_forward

        stage.forward = make_wrapper(original_forward, stage_idx)


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def ev_repr_to_img(x: np.ndarray) -> np.ndarray:
    """Convert stacked-histogram event representation to RGB."""
    ch, ht, wd = x.shape[-3:]
    assert ch > 1 and ch % 2 == 0
    ev_repr_reshaped = rearrange(x, '(posneg C) H W -> posneg C H W', posneg=2)
    img_neg = np.asarray(reduce(ev_repr_reshaped[0], 'C H W -> H W', 'sum'), dtype='int32')
    img_pos = np.asarray(reduce(ev_repr_reshaped[1], 'C H W -> H W', 'sum'), dtype='int32')
    img_diff = img_pos - img_neg
    img = 127 * np.ones((ht, wd, 3), dtype=np.uint8)
    img[img_diff > 0] = 255
    img[img_diff < 0] = 0
    return img


def spike_spatial_heatmap(spike: torch.Tensor) -> np.ndarray:
    """Convert (B,C,H,W) spike tensor to (H,W) firing rate heatmap [0,1]."""
    rate = spike[0].float().mean(dim=0).numpy()  # (H, W)
    vmax = rate.max()
    if vmax > 0:
        rate = rate / vmax
    return rate.astype(np.float32)


def overlay_heatmap(img: np.ndarray, heatmap: np.ndarray, alpha: float = 0.6) -> np.ndarray:
    """Overlay heatmap on RGB image."""
    h, w = img.shape[:2]
    hm_resized = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)
    hm_uint8 = (hm_resized * 255).astype(np.uint8)
    hm_color = cv2.applyColorMap(hm_uint8, cv2.COLORMAP_JET)
    hm_rgb = cv2.cvtColor(hm_color, cv2.COLOR_BGR2RGB)
    blended = alpha * hm_rgb.astype(np.float32) + (1 - alpha) * img.astype(np.float32)
    return np.clip(blended, 0, 255).astype(np.uint8)


def plot_spike_rasters(capture: SpikeCaptureHook, out_path: Path,
                       num_stages: int = 4, num_channels: int = 16,
                       spatial_pos: str = 'center'):
    """Plot spike raster: channels x timesteps for each stage.

    Shows output spikes and feedback (prev_spike) spikes side by side.
    """
    T = len(capture.timesteps)
    if T == 0:
        return

    fig, axes = plt.subplots(num_stages, 2, figsize=(16, 3 * num_stages),
                             squeeze=False)
    fig.suptitle(f'Spike Raster Plots ({T} timesteps)', fontsize=14)

    for si in range(num_stages):
        out_traces = []  # (T, C)
        fb_traces = []

        for t in range(T):
            if si not in capture.timesteps[t]:
                continue
            data = capture.timesteps[t][si]
            out_s = data['output_spike'][0]  # (C, H, W)
            fb_s = data['prev_spike']

            C, H, W = out_s.shape
            if spatial_pos == 'center':
                h, w = H // 2, W // 2
            else:
                h, w = 0, 0

            out_traces.append(out_s[:, h, w].numpy())
            if fb_s is not None:
                fb_traces.append(fb_s[0, :, h, w].numpy())
            else:
                fb_traces.append(np.zeros(C, dtype=np.float32))

        if not out_traces:
            continue

        out_arr = np.stack(out_traces, axis=0)  # (T, C)
        fb_arr = np.stack(fb_traces, axis=0)

        C_total = out_arr.shape[1]
        ch_indices = np.linspace(0, C_total - 1, min(num_channels, C_total), dtype=int)

        # Output spike raster
        ax = axes[si, 0]
        for idx, ch in enumerate(ch_indices):
            spike_times = np.where(out_arr[:, ch] > 0)[0]
            ax.scatter(spike_times, np.full_like(spike_times, idx),
                      marker='|', s=30, c='black', linewidths=0.8)
        ax.set_ylabel(f'Stage {si+1}\nChannel')
        ax.set_yticks(range(len(ch_indices)))
        ax.set_yticklabels([str(c) for c in ch_indices], fontsize=7)
        ax.set_xlim(-0.5, T - 0.5)
        if si == 0:
            ax.set_title('Output Spikes')
        if si == num_stages - 1:
            ax.set_xlabel('Timestep')

        # Feedback spike raster
        ax = axes[si, 1]
        for idx, ch in enumerate(ch_indices):
            spike_times = np.where(fb_arr[:, ch] > 0)[0]
            ax.scatter(spike_times, np.full_like(spike_times, idx),
                      marker='|', s=30, c='red', linewidths=0.8)
        ax.set_yticks(range(len(ch_indices)))
        ax.set_yticklabels([str(c) for c in ch_indices], fontsize=7)
        ax.set_xlim(-0.5, T - 0.5)
        if si == 0:
            ax.set_title('Feedback Spikes (prev_spike)')
        if si == num_stages - 1:
            ax.set_xlabel('Timestep')

    plt.tight_layout()
    plt.savefig(str(out_path / 'spike_raster.png'), dpi=150)
    plt.close()
    print(f'Saved spike raster to {out_path / "spike_raster.png"}')


def plot_firing_rates(capture: SpikeCaptureHook, out_path: Path,
                      num_stages: int = 4):
    """Plot mean firing rate over time for output and feedback spikes."""
    T = len(capture.timesteps)
    if T == 0:
        return

    fig, axes = plt.subplots(num_stages, 1, figsize=(12, 3 * num_stages),
                             squeeze=False)
    fig.suptitle(f'Mean Firing Rate Over Time ({T} timesteps)', fontsize=14)

    for si in range(num_stages):
        out_rates = []
        fb_rates = []

        for t in range(T):
            if si not in capture.timesteps[t]:
                out_rates.append(0)
                fb_rates.append(0)
                continue
            data = capture.timesteps[t][si]
            out_rates.append(data['output_spike'][0].float().mean().item())
            if data['prev_spike'] is not None:
                fb_rates.append(data['prev_spike'][0].float().mean().item())
            else:
                fb_rates.append(0.0)

        ax = axes[si, 0]
        ts = range(T)
        ax.plot(ts, out_rates, 'b-', label='Output spike rate', alpha=0.8)
        ax.plot(ts, fb_rates, 'r-', label='Feedback spike rate', alpha=0.8)
        ax.set_ylabel(f'Stage {si+1}\nRate')
        ax.legend(fontsize=8, loc='upper right')
        ax.set_xlim(0, T - 1)
        if si == num_stages - 1:
            ax.set_xlabel('Timestep')

    plt.tight_layout()
    plt.savefig(str(out_path / 'firing_rates.png'), dpi=150)
    plt.close()
    print(f'Saved firing rates to {out_path / "firing_rates.png"}')


def plot_spatial_heatmaps(capture: SpikeCaptureHook, ev_tensors: List[np.ndarray],
                          out_path: Path, num_stages: int = 4,
                          timesteps_to_show: Optional[List[int]] = None):
    """Plot spatial heatmaps of output and feedback spikes overlaid on events."""
    T = len(capture.timesteps)
    if T == 0:
        return

    if timesteps_to_show is None:
        timesteps_to_show = sorted(set([
            0, T // 4, T // 2, 3 * T // 4, T - 1
        ]))
    timesteps_to_show = [t for t in timesteps_to_show if t < T]

    n_ts = len(timesteps_to_show)
    fig, axes = plt.subplots(num_stages * 2, n_ts,
                             figsize=(4 * n_ts, 3 * num_stages * 2),
                             squeeze=False)
    fig.suptitle('Spatial Spike Heatmaps (top: output, bottom: feedback per stage)',
                 fontsize=14)

    for col, t in enumerate(timesteps_to_show):
        ev_img = ev_repr_to_img(ev_tensors[t]) if t < len(ev_tensors) else \
                 np.full((64, 80, 3), 127, dtype=np.uint8)

        for si in range(num_stages):
            row_out = si * 2
            row_fb = si * 2 + 1

            if si in capture.timesteps[t]:
                data = capture.timesteps[t][si]
                out_hm = spike_spatial_heatmap(data['output_spike'])
                out_img = overlay_heatmap(ev_img, out_hm, alpha=0.6)

                if data['prev_spike'] is not None:
                    fb_hm = spike_spatial_heatmap(data['prev_spike'])
                    fb_img = overlay_heatmap(ev_img, fb_hm, alpha=0.6)
                else:
                    fb_img = ev_img
            else:
                out_img = ev_img
                fb_img = ev_img

            axes[row_out, col].imshow(out_img)
            axes[row_out, col].axis('off')
            if col == 0:
                axes[row_out, col].set_ylabel(f'S{si+1} out', fontsize=10)

            axes[row_fb, col].imshow(fb_img)
            axes[row_fb, col].axis('off')
            if col == 0:
                axes[row_fb, col].set_ylabel(f'S{si+1} fb', fontsize=10)

        axes[0, col].set_title(f't={t}', fontsize=10)

    plt.tight_layout()
    plt.savefig(str(out_path / 'spatial_heatmaps.png'), dpi=150)
    plt.close()
    print(f'Saved spatial heatmaps to {out_path / "spatial_heatmaps.png"}')


def print_summary_stats(capture: SpikeCaptureHook, num_stages: int = 4):
    """Print summary statistics about spike activity."""
    T = len(capture.timesteps)
    print(f'\n{"="*60}')
    print(f'SPIKE DIAGNOSTIC SUMMARY ({T} timesteps)')
    print(f'{"="*60}')

    for si in range(num_stages):
        out_rates = []
        fb_rates = []

        for t in range(T):
            if si not in capture.timesteps[t]:
                continue
            data = capture.timesteps[t][si]
            out_rates.append(data['output_spike'][0].float().mean().item())
            if data['prev_spike'] is not None:
                fb_rates.append(data['prev_spike'][0].float().mean().item())

        if not out_rates:
            print(f'\nStage {si+1}: NO DATA')
            continue

        out_arr = np.array(out_rates)

        print(f'\nStage {si+1}:')
        print(f'  Output spike rate: mean={out_arr.mean():.4f}, '
              f'std={out_arr.std():.4f}, min={out_arr.min():.4f}, max={out_arr.max():.4f}')
        if fb_rates:
            fb_arr = np.array(fb_rates)
            print(f'  FB spike rate:     mean={fb_arr.mean():.4f}, '
                  f'std={fb_arr.std():.4f}, min={fb_arr.min():.4f}, max={fb_arr.max():.4f}')

            # With direct feedback (no fb_lif), output and feedback should match
            # (feedback IS the previous timestep's output spike)
            if len(fb_arr) > 1:
                corr = np.corrcoef(out_arr[1:len(fb_arr)+1], fb_arr[:len(out_arr)-1])[0, 1]
                if not np.isnan(corr):
                    print(f'  Out[t] vs FB[t] correlation: {corr:.4f} '
                          f'(should be ~1.0 with direct feedback)')
        else:
            print(f'  FB spike rate:     N/A (first timestep)')

        # Check for degenerate patterns
        if out_arr.mean() < 1e-6:
            print(f'  WARNING: Output spikes are essentially zero - stage is dead')
        if out_arr.mean() > 0.8:
            print(f'  WARNING: Output spike rate very high - possible saturation')

    print(f'{"="*60}\n')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(config_path='config', config_name='val', version_base='1.2')
def main(config: DictConfig):
    max_batches = int(config.get('max_batches', 5))
    output_dir = str(config.get('output_dir', './spike_diagnostics'))

    dynamically_modify_train_config(config)

    print('------ Configuration ------')
    print(OmegaConf.to_yaml(config))
    print('---------------------------')

    # GPU
    gpu_id = config.hardware.gpus
    assert isinstance(gpu_id, int), 'Only single-GPU supported'
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')

    # Data
    with open_dict(config):
        if 'train' not in config.batch_size:
            config.batch_size.train = config.batch_size.eval
        if 'train' not in config.hardware.num_workers:
            config.hardware.num_workers.train = config.hardware.num_workers.eval
    data_module = fetch_data_module(config=config)
    data_module.setup(stage='validate')
    val_loader = data_module.val_dataloader()

    # Model
    ckpt_path = Path(config.checkpoint)
    module = DetectionModule.load_from_checkpoint(str(ckpt_path), full_config=config)
    module = module.to(device)
    module.eval()
    mdl = module.mdl
    mdl_config = module.mdl_config
    in_res_hw = tuple(mdl_config.backbone.in_res_hw)
    input_padder = InputPadderFromShape(desired_hw=in_res_hw)
    rnn_states = RNNStates()

    num_stages = len(mdl.backbone.stages)

    # Patch backbone
    capture = SpikeCaptureHook()
    patch_backbone_for_capture(mdl.backbone, capture)
    print(f'Patched {num_stages} stages for spike capture')

    # Output dir
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = Path(output_dir) / f'diag_{timestamp}'
    out_path.mkdir(parents=True, exist_ok=True)
    print(f'Output directory: {out_path}')

    # Run inference and collect spikes
    ev_images_for_viz = []

    with torch.inference_mode():
        for batch_idx, batch in enumerate(val_loader):
            if batch_idx >= max_batches:
                break

            data = batch['data']
            worker_id = batch['worker_id']
            ev_tensor_sequence = data[DataType.EV_REPR]
            is_first_sample = data[DataType.IS_FIRST_SAMPLE]

            rnn_states.reset(worker_id=worker_id, indices_or_bool_tensor=is_first_sample)
            prev_states = rnn_states.get_states(worker_id=worker_id)

            sequence_len = len(ev_tensor_sequence)
            for tidx in range(sequence_len):
                capture.new_timestep()

                ev_tensor = ev_tensor_sequence[tidx].to(device=device, dtype=torch.float32)
                ev_tensor_padded = input_padder.pad_tensor_ev_repr(ev_tensor)

                backbone_features, states = mdl.forward_backbone(
                    x=ev_tensor_padded, previous_states=prev_states)
                prev_states = states

                ev_images_for_viz.append(ev_tensor_sequence[tidx][0].cpu().numpy())

            rnn_states.save_states_and_detach(worker_id=worker_id, states=prev_states)

            print(f'Batch {batch_idx + 1}/{max_batches} done '
                  f'({sequence_len} timesteps, total={len(capture.timesteps) + 1})')

    capture.finalize()
    print(f'\nTotal timesteps captured: {len(capture.timesteps)}')

    # Generate diagnostics
    print_summary_stats(capture, num_stages)
    plot_firing_rates(capture, out_path, num_stages)
    plot_spike_rasters(capture, out_path, num_stages)
    plot_spatial_heatmaps(capture, ev_images_for_viz, out_path, num_stages)

    print(f'\nAll diagnostics saved to {out_path}')


if __name__ == '__main__':
    main()
