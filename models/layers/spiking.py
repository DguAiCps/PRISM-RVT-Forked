"""Spiking neuron layers for SNN backbone."""
from typing import Optional, Tuple

import torch
import torch.nn as nn

from .pelif_spiking import _atan_surrogate, _triangle_surrogate


class LIFNeuron(nn.Module):
    """Leaky Integrate-and-Fire neuron (pure PyTorch).

    Membrane update:  V[t] = beta * V[t-1] + I[t]
    Spike:            S[t] = Heaviside(V[t] - threshold)   (surrogate grad)
    Reset:            V[t] = V[t] - S[t].detach() * threshold
    """

    def __init__(self,
                 beta_init: float = 0.9,
                 learn_beta: bool = True,
                 threshold: float = 1.0,
                 reset_mechanism: str = 'subtract',
                 channels: Optional[int] = None,
                 beta_spread: float = 0.0,
                 surrogate: str = 'triangle',
                 surrogate_alpha: float = 2.0,
                 surrogate_gamma: float = 1.0):
        super().__init__()
        assert reset_mechanism in ('subtract', 'zero'), \
            f"reset_mechanism must be 'subtract' or 'zero', got '{reset_mechanism}'"

        # Beta: per-channel if channels is set, otherwise scalar
        if channels is not None:
            beta_tensor = torch.empty(1, channels, 1, 1)
            nn.init.uniform_(beta_tensor, beta_init - beta_spread, beta_init + beta_spread)
            beta_tensor.clamp_(0.0, 1.0)
        else:
            beta_tensor = torch.tensor(beta_init)
        if learn_beta:
            self.beta = nn.Parameter(beta_tensor)
        else:
            self.register_buffer('beta', beta_tensor)

        self.threshold = threshold
        self.reset_mechanism = reset_mechanism

        # Surrogate function
        if surrogate == 'atan':
            self._spike_fn = lambda x: _atan_surrogate(x, surrogate_alpha)
        elif surrogate == 'triangle':
            self._spike_fn = lambda x: _triangle_surrogate(x, surrogate_gamma)
        else:
            raise ValueError(f"Unknown surrogate: {surrogate}. Use 'atan' or 'triangle'.")

    def forward(self,
                cur: torch.Tensor,
                mem: Optional[torch.Tensor] = None,
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        beta = self.beta.clamp(0.0, 1.0)

        if mem is None:
            mem = torch.zeros_like(cur)

        # Leaky integration
        mem = beta * mem + cur

        # Spike with surrogate gradient
        spike = self._spike_fn(mem - self.threshold)

        # Reset (not detached, matching PeLIF design)
        if self.reset_mechanism == 'subtract':
            mem = mem - spike * self.threshold
        else:  # 'zero'
            mem = mem * (1 - spike)

        return spike, mem


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
                 reset_mechanism: str = 'subtract',
                 channelwise_beta: bool = False,
                 beta_spread: float = 0.0,
                 surrogate: str = 'triangle'):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.lif = LIFNeuron(
            beta_init=beta_init,
            learn_beta=learn_beta,
            threshold=threshold,
            channels=out_channels if channelwise_beta else None,
            beta_spread=beta_spread,
            surrogate=surrogate,
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
        spike, mem = self.lif(cur, mem)
        return spike, mem
