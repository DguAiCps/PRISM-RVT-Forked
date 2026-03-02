"""Spiking neuron layers for SNN backbone."""
from typing import Optional, Tuple

import torch
import torch.nn as nn
import snntorch as snn


class SpikingConvBlock(nn.Module):
    """Conv2d + BatchNorm2d + LIF neuron.

    Input:  (N, C_in, H, W) float tensor
    Output: spike (N, C_out, H', W') binary, membrane (N, C_out, H', W') float
    State:  membrane potential V_t of shape (N, C_out, H', W')
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 padding: int = 1,
                 beta_init: float = 0.9,
                 learn_beta: bool = True,
                 threshold: float = 1.0,
                 reset_mechanism: str = 'subtract'):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.lif = snn.Leaky(
            beta=beta_init,
            learn_beta=learn_beta,
            threshold=threshold,
            reset_mechanism=reset_mechanism,
        )

    def forward(self,
                x: torch.Tensor,
                mem: Optional[torch.Tensor] = None,
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: input tensor (N, C_in, H, W)
            mem: previous membrane potential (N, C_out, H', W') or None
        Returns:
            spike: (N, C_out, H', W') binary
            mem: (N, C_out, H', W') float (new membrane potential)
        """
        cur = self.bn(self.conv(x))
        if mem is None:
            spike, mem = self.lif(cur)
        else:
            spike, mem = self.lif(cur, mem)
        return spike, mem
