from omegaconf import DictConfig

from .maxvit_rnn import RNNDetector as MaxViTRNNDetector
from .snn_cnn import SNNCNNBackbone, SNNCNNRecurrentBackbone, PlateauSNNCNNBackbone, SNNCNNLSTMBackbone
from .snn_swin import SNNSwinBackbone
from .softmax_swin import SoftmaxSwinBackbone


def build_recurrent_backbone(backbone_cfg: DictConfig):
    name = backbone_cfg.name
    if name == 'MaxViTRNN':
        return MaxViTRNNDetector(backbone_cfg)
    elif name == 'SNNCNN':
        return SNNCNNBackbone(backbone_cfg)
    elif name == 'SNNCNNRecurrent':
        return SNNCNNRecurrentBackbone(backbone_cfg)
    elif name == 'SNNSwin':
        return SNNSwinBackbone(backbone_cfg)
    elif name == 'SoftmaxSwin':
        return SoftmaxSwinBackbone(backbone_cfg)
    elif name == 'PlateauSNNCNN':
        return PlateauSNNCNNBackbone(backbone_cfg)
    elif name == 'SNNCNNLSTM':
        return SNNCNNLSTMBackbone(backbone_cfg)
    else:
        raise NotImplementedError
