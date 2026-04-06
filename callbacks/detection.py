from enum import Enum, auto
from typing import Any

import cv2
import numpy as np
import pytorch_lightning as pl
import torch
from einops import rearrange
from omegaconf import DictConfig

from data.utils.types import ObjDetOutput
from loggers.wandb_logger import WandbLogger
from models.detection.recurrent_backbone.snn_swin import window_reverse
from utils.evaluation.prophesee.visualize.vis_utils import LABELMAP_GEN1, LABELMAP_GEN4_SHORT, draw_bboxes
from .viz_base import VizCallbackBase


class DetectionVizEnum(Enum):
    EV_IMG = auto()
    LABEL_IMG_PROPH = auto()
    PRED_IMG_PROPH = auto()


def _collect_attn_heatmap(backbone, stage_idx: int = 1) -> np.ndarray | None:
    """Collect stored attention weights from a backbone stage and produce a spatial heatmap.

    Args:
        backbone: SNNSwinBackbone instance.
        stage_idx: 0-indexed stage to visualize.
    Returns:
        (H, W) float32 heatmap in [0, 1], or None if no attention stored.
    """
    stage = backbone.stages[stage_idx]
    if not hasattr(stage, 'blocks'):
        return None
    spatial_maps = []

    for blk in stage.blocks:
        attn_mod = blk.attn
        if not hasattr(attn_mod, '_last_attn') or attn_mod._last_attn is None:
            continue
        attn = attn_mod._last_attn.cpu()  # (BnW, H_heads, N, N)
        meta = attn_mod._last_attn_meta
        B = meta['B']
        nW = meta['nW']
        N = meta['N']
        H_heads = meta['H_heads']
        H, W = blk.input_resolution
        ws = blk.window_size
        shift = blk.shift_size
        global_attn = blk.global_attn

        # Use batch index 0
        attn = attn.reshape(B, nW, H_heads, N, N)[0]  # (nW, H_heads, N, N)
        attn_avg = attn.mean(dim=1)  # (nW, N, N)
        key_importance = attn_avg.mean(dim=1)  # (nW, N)

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

    if not spatial_maps:
        return None

    combined = np.mean(spatial_maps, axis=0)
    vmin, vmax = combined.min(), combined.max()
    if vmax - vmin > 1e-8:
        combined = (combined - vmin) / (vmax - vmin)
    else:
        combined = np.zeros_like(combined)
    return combined.astype(np.float32)


def _overlay_heatmap(img: np.ndarray, heatmap: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Overlay a heatmap on an RGB image."""
    h, w = img.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)
    heatmap_uint8 = (heatmap_resized * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    blended = (alpha * heatmap_rgb.astype(np.float32) +
               (1 - alpha) * img.astype(np.float32))
    return np.clip(blended, 0, 255).astype(np.uint8)


def _collect_snn_scalars(backbone) -> dict:
    """Collect learnable SNN parameters (beta, threshold, etc.) from backbone stages."""
    scalars = {}
    if backbone is None or not hasattr(backbone, 'stages'):
        return scalars
    for si, stage in enumerate(backbone.stages):
        # SpikingConvBlock layers (standard LIF)
        layers = getattr(stage, 'layers', [])
        for li, layer in enumerate(layers):
            lif = getattr(layer, 'lif', None)
            if lif is None:
                continue
            if hasattr(lif, 'beta'):
                beta = lif.beta.clamp(0.0, 1.0)
                prefix = f'snn/stage{si}_layer{li}_beta'
                if beta.numel() == 1:
                    scalars[prefix] = beta.item()
                else:
                    scalars[f'{prefix}_mean'] = beta.mean().item()
                    scalars[f'{prefix}_std'] = beta.std().item()
                    scalars[f'{prefix}_min'] = beta.min().item()
                    scalars[f'{prefix}_max'] = beta.max().item()
            if hasattr(lif, 'reset_ratio'):
                rr = lif.reset_ratio.clamp(0.0, 1.0)
                prefix = f'snn/stage{si}_layer{li}_reset_ratio'
                if rr.numel() == 1:
                    scalars[prefix] = rr.item()
                else:
                    scalars[f'{prefix}_mean'] = rr.mean().item()
                    scalars[f'{prefix}_std'] = rr.std().item()
                    scalars[f'{prefix}_min'] = rr.min().item()
                    scalars[f'{prefix}_max'] = rr.max().item()
        # PlateauSpikingConvBlock (last layer in PlateauSNNCNNStage)
        plateau = getattr(stage, 'plateau_layer', None)
        if plateau is not None:
            plif = plateau.plateau_lif
            gate_lif = plif.gate_lif
            scalars[f'snn/stage{si}_gate_beta'] = gate_lif.beta.clamp(0.0, 1.0).item()
            scalars[f'snn/stage{si}_gate_tonic'] = plif.gate_tonic.item()
        # Feedback LIF (SNNCNNRecurrentStage)
        fb_lif = getattr(stage, 'fb_lif', None)
        if fb_lif is not None and hasattr(fb_lif, 'beta'):
            scalars[f'snn/stage{si}_fb_beta'] = fb_lif.beta.clamp(0.0, 1.0).item()
    return scalars


class DetectionVizCallback(VizCallbackBase):
    def __init__(self, config: DictConfig):
        super().__init__(config=config, buffer_entries=DetectionVizEnum)

        dataset_name = config.dataset.name
        if dataset_name == 'gen1':
            self.label_map = LABELMAP_GEN1
        elif dataset_name == 'gen4':
            self.label_map = LABELMAP_GEN4_SHORT
        else:
            raise NotImplementedError

    def on_train_batch_end(
            self,
            trainer: pl.Trainer,
            pl_module: pl.LightningModule,
            outputs: Any,
            batch: Any,
            batch_idx: int,
            unused: int = 0,
    ) -> None:
        # Log SNN learnable params every step (actual wandb write freq
        # controlled by trainer's log_every_n_steps, default 50)
        backbone = getattr(getattr(pl_module, 'mdl', None), 'backbone', None)
        snn_scalars = _collect_snn_scalars(backbone)
        if snn_scalars:
            pl_module.log_dict(snn_scalars, on_step=True, on_epoch=False)

        # Delegate to base class for high-dim visualization
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx, unused)

    def on_train_batch_end_custom(self,
                                  logger: WandbLogger,
                                  pl_module: pl.LightningModule,
                                  outputs: Any,
                                  batch: Any,
                                  log_n_samples: int,
                                  global_step: int) -> None:
        if outputs is None:
            return
        ev_tensors = outputs[ObjDetOutput.EV_REPR]
        num_samples = len(ev_tensors)
        assert num_samples > 0
        log_n_samples = min(num_samples, log_n_samples)

        # Collect attention heatmaps from all stages
        attn_heatmaps = {}
        backbone = getattr(getattr(pl_module, 'mdl', None), 'backbone', None)
        if backbone is not None and hasattr(backbone, 'stages'):
            for si in range(len(backbone.stages)):
                hm = _collect_attn_heatmap(backbone, stage_idx=si)
                if hm is not None:
                    attn_heatmaps[si] = hm

        merged_img = []
        captions = []
        start_idx = num_samples - 1
        end_idx = start_idx - log_n_samples
        for sample_idx in range(start_idx, end_idx, -1):
            ev_img = self.ev_repr_to_img(ev_tensors[sample_idx].cpu().numpy())

            predictions_proph = outputs[ObjDetOutput.PRED_PROPH][sample_idx]
            prediction_img = ev_img.copy()
            draw_bboxes(prediction_img, predictions_proph, labelmap=self.label_map)

            labels_proph = outputs[ObjDetOutput.LABELS_PROPH][sample_idx]
            label_img = ev_img.copy()
            draw_bboxes(label_img, labels_proph, labelmap=self.label_map)

            merged_img.append(rearrange([prediction_img, label_img], 'pl H W C -> (pl H) W C', pl=2, C=3))
            captions.append(f'sample_{sample_idx}')

        logger.log_images(key='train/predictions',
                          images=merged_img,
                          caption=captions,
                          step=global_step)

        # Log attention heatmaps as separate images (one per stage)
        if attn_heatmaps:
            ev_img_for_attn = self.ev_repr_to_img(ev_tensors[start_idx].cpu().numpy())
            attn_images = []
            attn_captions = []
            for si in sorted(attn_heatmaps.keys()):
                attn_img = _overlay_heatmap(ev_img_for_attn, attn_heatmaps[si], alpha=0.6)
                attn_images.append(attn_img)
                attn_captions.append(f'stage_{si + 1}')
            logger.log_images(key='train/attention',
                              images=attn_images,
                              caption=attn_captions,
                              step=global_step)

    def on_validation_batch_end_custom(self, batch: Any, outputs: Any):
        if outputs[ObjDetOutput.SKIP_VIZ]:
            return
        ev_tensor = outputs[ObjDetOutput.EV_REPR]
        assert isinstance(ev_tensor, torch.Tensor)

        ev_img = self.ev_repr_to_img(ev_tensor.cpu().numpy())

        predictions_proph = outputs[ObjDetOutput.PRED_PROPH]
        prediction_img = ev_img.copy()
        draw_bboxes(prediction_img, predictions_proph, labelmap=self.label_map)
        self.add_to_buffer(DetectionVizEnum.PRED_IMG_PROPH, prediction_img)

        labels_proph = outputs[ObjDetOutput.LABELS_PROPH]
        label_img = ev_img.copy()
        draw_bboxes(label_img, labels_proph, labelmap=self.label_map)
        self.add_to_buffer(DetectionVizEnum.LABEL_IMG_PROPH, label_img)

    def on_validation_epoch_end_custom(self, logger: WandbLogger):
        pred_imgs = self.get_from_buffer(DetectionVizEnum.PRED_IMG_PROPH)
        label_imgs = self.get_from_buffer(DetectionVizEnum.LABEL_IMG_PROPH)
        assert len(pred_imgs) == len(label_imgs)
        merged_img = []
        captions = []
        for idx, (pred_img, label_img) in enumerate(zip(pred_imgs, label_imgs)):
            merged_img.append(rearrange([pred_img, label_img], 'pl H W C -> (pl H) W C', pl=2, C=3))
            captions.append(f'sample_{idx}')

        logger.log_images(key='val/predictions',
                          images=merged_img,
                          caption=captions)
