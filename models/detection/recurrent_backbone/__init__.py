from omegaconf import DictConfig

from .maxvit_rnn import RNNDetector as MaxViTRNNDetector
from .snn_cnn import SNNCNNBackbone
from .snn_swin import SNNSwinBackbone
from .pelif_cnn import PeLIFCNNBackbone
from .pelif_cnn_attn import PeLIFCNNAttnBackbone


def build_recurrent_backbone(backbone_cfg: DictConfig):
    name = backbone_cfg.name
    if name == 'MaxViTRNN':
        return MaxViTRNNDetector(backbone_cfg)
    elif name == 'SNNCNN':
        return SNNCNNBackbone(backbone_cfg)
    elif name == 'SNNSwin':
        return SNNSwinBackbone(backbone_cfg)
    elif name == 'PeLIFCNN':
        return PeLIFCNNBackbone(backbone_cfg)
    elif name == 'PeLIFCNNAttn':
        return PeLIFCNNAttnBackbone(backbone_cfg)
    else:
        raise NotImplementedError
