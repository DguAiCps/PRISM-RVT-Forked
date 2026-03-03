from omegaconf import DictConfig

from .maxvit_rnn import RNNDetector as MaxViTRNNDetector
from .snn_cnn import SNNCNNBackbone
from .snn_swin import SNNSwinBackbone


def build_recurrent_backbone(backbone_cfg: DictConfig):
    name = backbone_cfg.name
    if name == 'MaxViTRNN':
        return MaxViTRNNDetector(backbone_cfg)
    elif name == 'SNNCNN':
        return SNNCNNBackbone(backbone_cfg)
    elif name == 'SNNSwin':
        return SNNSwinBackbone(backbone_cfg)
    else:
        raise NotImplementedError
