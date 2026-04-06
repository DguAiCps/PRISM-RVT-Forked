"""Streaming detection visualization script.

Usage
-----
# With model predictions (side-by-side: predictions | ground truth):
python visualize.py dataset=gen1 +model/snn_swin_yolox=default \
    'checkpoint="/path/to/ckpt.ckpt"' dataset.path=/path/to/gen1 \
    hardware.gpus=0 batch_size.eval=1 hardware.num_workers.eval=0 \
    +save_video=true

# Data-only mode (event frames + GT labels, no model needed):
python visualize.py dataset=gen1 dataset.path=/path/to/gen1 \
    batch_size.eval=1 hardware.num_workers.eval=0 \
    +data_only=true +save_video=true

# Attention visualization (heatmap overlay of what the model attends to):
python visualize.py dataset=gen1 +model/snn_swin_yolox=default \
    'checkpoint="/path/to/ckpt.ckpt"' dataset.path=/path/to/gen1 \
    hardware.gpus=0 batch_size.eval=1 hardware.num_workers.eval=0 \
    +attn_vis=true +save_video=true +attn_stage=2

Optional Hydra overrides:
    +save_video=true        Save an MP4 video in addition to frames
    +max_frames=500         Stop after N frames (default: unlimited)
    +output_dir=./viz_out   Custom output directory
    +fps=20                 Video FPS (default: 20)
    +data_only=true         Skip model, just visualize data + GT labels
    +attn_vis=true          Overlay attention heatmaps on event frames
    +attn_stage=2           Which stage to visualize (1-4, default: 2)
"""
import os

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

from datetime import datetime
from pathlib import Path

import cv2
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
from models.detection.yolox.utils.boxes import postprocess
from modules.utils.fetch import fetch_data_module
from modules.detection import Module as DetectionModule
from utils.evaluation.prophesee.io.box_loading import to_prophesee, loaded_label_to_prophesee, BBOX_DTYPE
from utils.evaluation.prophesee.visualize.vis_utils import (
    draw_bboxes_bbv,
    LABELMAP_GEN1,
    LABELMAP_GEN4_SHORT,
)
from utils.padding import InputPadderFromShape
from modules.utils.detection import RNNStates
from models.detection.recurrent_backbone.snn_swin import (
    SSAWindowAttention, window_reverse,
)


class AttentionCaptureHook:
    """Captures attention weight maps from SSAWindowAttention."""

    def __init__(self):
        self.attn_weights = []  # continuous attention: list of (BnW, H_heads, N, N)
        self.metadata = []

    def clear(self):
        self.attn_weights.clear()
        self.metadata.clear()


def register_attn_hooks(backbone, target_stage: int):
    """Register hooks on SSAWindowAttention modules in the target stage.

    Monkey-patches the forward method to capture the Q@K^T attention weights.
    """
    capture = AttentionCaptureHook()
    stage = backbone.stages[target_stage - 1]  # 1-indexed

    for blk_idx, blk in enumerate(stage.blocks):
        attn_module = blk.attn
        original_forward = attn_module.forward

        def make_wrapper(orig_fwd, mod, blk_ref):
            def wrapped_forward(x, prev_mems, B, mask=None):
                # Recompute Q@K^T to capture attention weights
                BnW, N, C = x.shape
                H_heads = mod.num_heads
                d = mod.head_dim
                nW = BnW // B

                # Unpack previous membranes
                if prev_mems is None:
                    prev_mems_list = [None] * mod.NUM_STATES
                else:
                    prev_mems_list = prev_mems

                p_q = prev_mems_list[0]
                p_k = prev_mems_list[1]
                if p_q is not None:
                    p_q = p_q.reshape(BnW, N, C)
                if p_k is not None:
                    p_k = p_k.reshape(BnW, N, C)

                # Compute Q and K spikes (same as forward)
                q_spike, _ = mod.lif_q(mod.bn_q(mod.proj_q(x)), p_q)
                q = q_spike.reshape(BnW, N, H_heads, d).permute(0, 2, 1, 3).contiguous()
                k_spike, _ = mod.lif_k(mod.bn_k(mod.proj_k(x)), p_k)
                k = k_spike.reshape(BnW, N, H_heads, d).permute(0, 2, 1, 3).contiguous()

                attn = (q @ k.transpose(-2, -1)) * mod.attn_scale
                if mod.use_rel_pos_bias:
                    attn = attn + mod._get_rel_pos_bias().unsqueeze(0)

                capture.attn_weights.append(attn.detach().cpu())
                capture.metadata.append({
                    'B': B, 'nW': nW, 'N': N, 'H_heads': H_heads,
                    'window_size': blk_ref.window_size,
                    'input_resolution': blk_ref.input_resolution,
                    'shift_size': blk_ref.shift_size,
                    'global_attn': blk_ref.global_attn,
                })

                # Run the actual forward
                return orig_fwd(x, prev_mems, B, mask)
            return wrapped_forward

        attn_module.forward = make_wrapper(original_forward, attn_module, blk)

    return capture


def attn_to_spatial_map(capture: AttentionCaptureHook, batch_idx: int = 0) -> np.ndarray:
    """Convert captured attention weights to a spatial heatmap.

    For each block, compute average attention each spatial position receives
    (mean over heads, mean over query dimension). Then average across blocks.

    Returns: (H, W) float32 heatmap normalized to [0, 1].
    """
    spatial_maps = []

    for blk_i in range(len(capture.attn_weights)):
        meta = capture.metadata[blk_i]
        B = meta['B']
        nW = meta['nW']
        N = meta['N']
        H_heads = meta['H_heads']
        ws = meta['window_size']
        H, W = meta['input_resolution']
        shift = meta['shift_size']
        global_attn = meta['global_attn']

        # attn_weights: (BnW, H_heads, N, N)
        attn = capture.attn_weights[blk_i]
        BnW = B * nW
        # Reshape to (B, nW, H_heads, N, N)
        attn = attn.reshape(B, nW, H_heads, N, N)
        attn = attn[batch_idx]  # (nW, H_heads, N, N)

        # Average over heads: (nW, N, N)
        attn_avg = attn.mean(dim=1)

        # For each key token, mean attention it receives from all queries: (nW, N)
        key_importance = attn_avg.mean(dim=1)

        if global_attn:
            spatial = key_importance[0].reshape(H, W)
        else:
            key_2d = key_importance.reshape(nW, ws, ws)
            key_4d = key_2d.unsqueeze(-1)
            spatial_4d = window_reverse(key_4d, ws, H, W)
            spatial = spatial_4d[0, :, :, 0]

            if shift > 0:
                spatial = torch.roll(spatial, shifts=(shift, shift), dims=(0, 1))

        spatial_maps.append(spatial.numpy())

    combined = np.mean(spatial_maps, axis=0)

    vmin, vmax = combined.min(), combined.max()
    if vmax - vmin > 1e-8:
        combined = (combined - vmin) / (vmax - vmin)
    else:
        combined = np.zeros_like(combined)

    return combined.astype(np.float32)


def overlay_heatmap(img: np.ndarray, heatmap: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Overlay a heatmap on an RGB image.

    Args:
        img: (H_img, W_img, 3) uint8 RGB image.
        heatmap: (H_feat, W_feat) float32 in [0, 1].
        alpha: blending factor.
    Returns:
        (H_img, W_img, 3) uint8 RGB image with heatmap overlay.
    """
    h, w = img.shape[:2]
    # Resize heatmap to image size
    heatmap_resized = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)
    # Apply colormap (JET: blue=cold, red=hot)
    heatmap_uint8 = (heatmap_resized * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    # Blend
    blended = (alpha * heatmap_rgb.astype(np.float32) +
               (1 - alpha) * img.astype(np.float32))
    return np.clip(blended, 0, 255).astype(np.uint8)


def ev_repr_to_img(x: np.ndarray) -> np.ndarray:
    """Convert stacked-histogram event representation to an RGB image."""
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


def write_frame(img_rgb, frame_idx, frames_path, video_writer, save_video, fps, out_path):
    """Save a frame as PNG and optionally write to video."""
    frame_file = frames_path / f'frame_{frame_idx:06d}.png'
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(frame_file), img_bgr)

    if save_video and video_writer[0] is None:
        h, w = img_rgb.shape[:2]
        video_file = str(out_path / 'output.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer[0] = cv2.VideoWriter(video_file, fourcc, fps, (w, h))
        print(f'Saving video to {video_file}')

    if video_writer[0] is not None:
        video_writer[0].write(img_bgr)


@hydra.main(config_path='config', config_name='val', version_base='1.2')
def main(config: DictConfig):
    # ---- Viz-specific overrides ----
    data_only = bool(config.get('data_only', False))
    save_video = bool(config.get('save_video', False))
    max_frames = int(config.get('max_frames', 0))  # 0 = unlimited
    output_dir = str(config.get('output_dir', './visualization_output'))
    fps = int(config.get('fps', 20))
    attn_vis = bool(config.get('attn_vis', False))
    attn_stage = int(config.get('attn_stage', 2))  # 1-indexed
    conf_thre = float(config.get('conf_thre', 0.0))  # 0 = use model default
    skip_batches = int(config.get('skip_batches', 0))  # skip N batches before recording
    stateless = bool(config.get('stateless', False))  # reset states every timestep (ablation: no temporal memory)

    # In data_only mode, fill in mandatory model config placeholders
    if data_only:
        with open_dict(config):
            config.model.backbone.name = 'SNNSwin'
            config.model.backbone.input_channels = 20
            config.model.backbone.embed_dim = 40
            config.model.backbone.depths = [2, 2, 2, 2]
            config.model.backbone.num_heads = [2, 4, 8, 8]
            config.model.backbone.window_sizes = [8, 8, 4, None]
            config.model.backbone.mlp_ratio = 4.0
            config.model.backbone.attn_scale = 0.125
            config.model.backbone.use_rel_pos_bias = False
            config.model.backbone.output_stages = [2, 3, 4]
            config.model.backbone.snn = {
                'beta_init': 0.5, 'learn_beta': True,
                'threshold': 1.0, 'reset_mechanism': 'subtract'}
            config.model.fpn.name = 'PAFPN'
            config.model.fpn.depth = 0.67
            config.model.fpn.in_stages = [2, 3, 4]
            config.model.fpn.depthwise = False
            config.model.fpn.act = 'silu'
            config.model.fpn.compile = {'enable': False, 'args': {'mode': 'reduce-overhead'}}
            config.model.head.name = 'YoloX'
            config.model.head.depthwise = False
            config.model.head.act = 'silu'
            config.model.head.num_classes = 2
            config.model.head.compile = {'enable': False, 'args': {'mode': 'reduce-overhead'}}
            if OmegaConf.is_missing(config, 'checkpoint'):
                config.checkpoint = 'none'

    dynamically_modify_train_config(config)

    print('------ Configuration ------')
    print(OmegaConf.to_yaml(config))
    print('---------------------------')

    # ---- GPU ----
    gpu_id = config.hardware.gpus
    assert isinstance(gpu_id, int), 'Only single-GPU supported'
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')

    # ---- Data ----
    with open_dict(config):
        if 'train' not in config.batch_size:
            config.batch_size.train = config.batch_size.eval
        if 'train' not in config.hardware.num_workers:
            config.hardware.num_workers.train = config.hardware.num_workers.eval
    data_module = fetch_data_module(config=config)
    data_module.setup(stage='validate')
    val_loader = data_module.val_dataloader()

    # ---- Label map ----
    dataset_name = config.dataset.name
    if dataset_name == 'gen1':
        label_map = LABELMAP_GEN1
    elif dataset_name == 'gen4':
        label_map = LABELMAP_GEN4_SHORT
    else:
        raise NotImplementedError(f'Unknown dataset: {dataset_name}')

    # ---- Model (only if not data_only) ----
    mdl = None
    mdl_config = None
    input_padder = None
    rnn_states = None
    attn_capture = None
    if not data_only:
        ckpt_path = Path(config.checkpoint)
        module = DetectionModule.load_from_checkpoint(str(ckpt_path), full_config=config)
        module = module.to(device)
        module.eval()
        mdl = module.mdl
        mdl_config = module.mdl_config
        in_res_hw = tuple(mdl_config.backbone.in_res_hw)
        input_padder = InputPadderFromShape(desired_hw=in_res_hw)
        rnn_states = RNNStates()

        if attn_vis:
            attn_capture = register_attn_hooks(mdl.backbone, target_stage=attn_stage)
            print(f'Attention visualization enabled for stage {attn_stage}')

    # ---- Output directory ----
    mode_str = 'data_only' if data_only else 'detection'
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = Path(output_dir) / f'{mode_str}_{timestamp}'
    frames_path = out_path / 'frames'
    frames_path.mkdir(parents=True, exist_ok=True)
    print(f'Saving frames to {frames_path}')

    # ---- Main loop ----
    frame_idx = 0
    video_writer = [None]  # mutable wrapper for write_frame helper
    eval_sampling_mode = DatasetSamplingMode(config.dataset.eval.sampling)
    done = False

    with torch.inference_mode():
        for batch_idx, batch in enumerate(val_loader):
            if done:
                break

            data = batch['data']
            worker_id = batch['worker_id']
            ev_tensor_sequence = data[DataType.EV_REPR]
            sparse_obj_labels = data[DataType.OBJLABELS_SEQ]
            is_first_sample = data[DataType.IS_FIRST_SAMPLE]

            # State management for model mode
            prev_states = None
            if rnn_states is not None:
                rnn_states.reset(worker_id=worker_id, indices_or_bool_tensor=is_first_sample)
                prev_states = rnn_states.get_states(worker_id=worker_id)

            sequence_len = len(ev_tensor_sequence)
            batch_size = ev_tensor_sequence[0].shape[0]
            recording = batch_idx >= skip_batches

            for tidx in range(sequence_len):
                if done:
                    break

                collect = (tidx == sequence_len - 1) or \
                          (eval_sampling_mode == DatasetSamplingMode.STREAM)

                # ===== Model inference path =====
                if not data_only:
                    ev_tensor = ev_tensor_sequence[tidx].to(device=device, dtype=torch.float32)
                    ev_tensor = input_padder.pad_tensor_ev_repr(ev_tensor)

                    backbone_features, states = mdl.forward_backbone(
                        x=ev_tensor, previous_states=None if stateless else prev_states)
                    prev_states = states

                    if not collect or not recording:
                        continue

                    # Run detection at every collected timestep
                    predictions, _ = mdl.forward_detect(backbone_features=backbone_features)
                    effective_conf = conf_thre if conf_thre > 0 else mdl_config.postprocess.confidence_threshold
                    pred_processed = postprocess(
                        prediction=predictions,
                        num_classes=mdl_config.head.num_classes,
                        conf_thre=effective_conf,
                        nms_thre=mdl_config.postprocess.nms_threshold)

                    # Check if GT labels exist at this timestep
                    current_labels, valid_batch_indices = \
                        sparse_obj_labels[tidx].get_valid_labels_and_batch_indices()
                    has_gt = len(current_labels) > 0

                    # Empty bbox array for draw_bboxes_bbv (ensures consistent scaling)
                    empty_boxes = np.zeros((0,), dtype=BBOX_DTYPE)

                    # Compute attention heatmap if enabled
                    attn_heatmaps = {}
                    if attn_capture is not None and len(attn_capture.attn_weights) > 0:
                        for bi in range(batch_size):
                            attn_heatmaps[bi] = attn_to_spatial_map(attn_capture, batch_idx=bi)
                        attn_capture.clear()

                    for bi in range(batch_size):
                        ev_np = ev_tensor_sequence[tidx][bi].cpu().numpy()
                        ev_img = ev_repr_to_img(ev_np)

                        # Predictions: always draw
                        pred_i = pred_processed[bi]
                        if pred_i is not None and pred_i.shape[0] > 0:
                            num_pred = pred_i.shape[0]
                            pred_proph = np.zeros((num_pred,), dtype=BBOX_DTYPE)
                            pred_np = pred_i.detach().cpu().numpy()
                            pred_proph['x'] = pred_np[:, 0].astype(BBOX_DTYPE['x'])
                            pred_proph['y'] = pred_np[:, 1].astype(BBOX_DTYPE['y'])
                            pred_proph['w'] = (pred_np[:, 2] - pred_np[:, 0]).astype(BBOX_DTYPE['w'])
                            pred_proph['h'] = (pred_np[:, 3] - pred_np[:, 1]).astype(BBOX_DTYPE['h'])
                            pred_proph['class_id'] = pred_np[:, 6].astype(BBOX_DTYPE['class_id'])
                            pred_proph['class_confidence'] = pred_np[:, 5].astype(BBOX_DTYPE['class_confidence'])
                        else:
                            pred_proph = empty_boxes
                        pred_img = draw_bboxes_bbv(ev_img.copy(), pred_proph, labelmap=label_map)

                        # GT: draw when available, otherwise pass empty boxes (keeps scale)
                        if has_gt and bi in valid_batch_indices:
                            li = valid_batch_indices.index(bi)
                            gt_proph = loaded_label_to_prophesee(current_labels[li])
                        else:
                            gt_proph = empty_boxes
                        label_img = draw_bboxes_bbv(ev_img.copy(), gt_proph, labelmap=label_map)

                        # Attention heatmap overlay
                        if bi in attn_heatmaps:
                            attn_img = overlay_heatmap(ev_img, attn_heatmaps[bi], alpha=0.6)
                            # Scale attn_img to match pred/label image size
                            attn_img = cv2.resize(attn_img, (pred_img.shape[1], pred_img.shape[0]),
                                                  interpolation=cv2.INTER_LINEAR)
                            # 3-panel: attention | predictions | ground truth
                            merged = np.concatenate([attn_img, pred_img, label_img], axis=1)
                            header_h = 30
                            header = np.zeros((header_h, merged.shape[1], 3), dtype=np.uint8)
                            panel_w = merged.shape[1] // 3
                            cv2.putText(header, f'Attention (stage {attn_stage})', (10, 20),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                            cv2.putText(header, 'Predictions', (panel_w + 10, 20),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                            gt_text = 'Ground Truth' if (has_gt and bi in valid_batch_indices) else 'GT (N/A)'
                            cv2.putText(header, gt_text, (2 * panel_w + 10, 20),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                        else:
                            # 2-panel: predictions | ground truth
                            merged = np.concatenate([pred_img, label_img], axis=1)
                            header_h = 30
                            header = np.zeros((header_h, merged.shape[1], 3), dtype=np.uint8)
                            mid = merged.shape[1] // 2
                            cv2.putText(header, 'Predictions', (10, 20),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                            gt_text = 'Ground Truth' if (has_gt and bi in valid_batch_indices) else 'GT (N/A)'
                            cv2.putText(header, gt_text, (mid + 10, 20),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                        merged = np.concatenate([header, merged], axis=0)

                        write_frame(merged, frame_idx, frames_path, video_writer,
                                    save_video, fps, out_path)
                        frame_idx += 1
                        if 0 < max_frames <= frame_idx:
                            done = True
                            break

                # ===== Data-only path =====
                else:
                    if not recording:
                        continue
                    for bi in range(batch_size):
                        ev_np = ev_tensor_sequence[tidx][bi].cpu().numpy()
                        ev_img = ev_repr_to_img(ev_np)

                        # Draw GT labels if available at this timestep
                        current_labels, valid_batch_indices = \
                            sparse_obj_labels[tidx].get_valid_labels_and_batch_indices()
                        if bi in valid_batch_indices:
                            li = valid_batch_indices.index(bi)
                            labels_proph = loaded_label_to_prophesee(current_labels[li])
                            out_img = draw_bboxes_bbv(ev_img, labels_proph, labelmap=label_map)
                        else:
                            out_img = ev_img

                        # Add header with frame info
                        header_h = 30
                        header = np.zeros((header_h, out_img.shape[1], 3), dtype=np.uint8)
                        has_gt = bi in valid_batch_indices
                        label_text = 'Events + GT' if has_gt else 'Events'
                        cv2.putText(header, f'{label_text}  (frame {frame_idx})', (10, 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        out_img = np.concatenate([header, out_img], axis=0)

                        write_frame(out_img, frame_idx, frames_path, video_writer,
                                    save_video, fps, out_path)
                        frame_idx += 1
                        if 0 < max_frames <= frame_idx:
                            done = True
                            break

            # Save states for model mode
            if rnn_states is not None and prev_states is not None:
                rnn_states.save_states_and_detach(worker_id=worker_id, states=prev_states)

            if batch_idx % 50 == 0:
                print(f'Batch {batch_idx} | Frames saved: {frame_idx}')

    if video_writer[0] is not None:
        video_writer[0].release()

    print(f'\nDone! Saved {frame_idx} frames to {frames_path}')
    if save_video:
        print(f'Video saved to {out_path / "output.mp4"}')


if __name__ == '__main__':
    main()
