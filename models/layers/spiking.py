"""Spiking neuron layers for SNN backbone."""
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


def _atan_surrogate(mem_shift: torch.Tensor, alpha: float = 2.0) -> torch.Tensor:
    """Spike function with ATan surrogate gradient (pure PyTorch, no custom autograd).

    Forward: Heaviside step  S = (mem_shift > 0).float()
    Backward: dS/d(mem_shift) = alpha/2 / (1 + (pi/2 * alpha * mem_shift)^2)

    Uses the straight-through estimator trick so that the forward output is
    binary but gradients flow through the smooth arctan surrogate.
    """
    smooth = (1.0 / math.pi) * torch.atan(
        (math.pi * alpha / 2.0) * mem_shift
    ) + 0.5
    binary = (mem_shift > 0).float()
    # Forward = binary,  Backward = d(smooth)/d(mem_shift)
    return smooth + (binary - smooth).detach()


class LIFNeuron(nn.Module):
    """Leaky Integrate-and-Fire neuron (pure PyTorch, replaces snntorch.Leaky).

    Implements the same dynamics as ``snntorch.Leaky`` with
    ``reset_mechanism='subtract'`` but uses only standard PyTorch ops,
    avoiding the in-place ``pow_()`` in snntorch's ATan surrogate backward.

    Membrane update:  V[t] = beta * V[t-1] + I[t]
    Spike:            S[t] = Heaviside(V[t] - threshold)   (ATan surrogate grad)
    Reset:            V[t] = V[t] - S[t].detach() * threshold
    """

    def __init__(self,
                 beta_init: float = 0.9,
                 learn_beta: bool = True,
                 threshold: float = 1.0,
                 alpha: float = 2.0,
                 reset_mechanism: str = 'subtract'):
        super().__init__()
        assert reset_mechanism in ('subtract', 'zero'), \
            f"reset_mechanism must be 'subtract' or 'zero', got '{reset_mechanism}'"
        beta_tensor = torch.tensor(beta_init)
        if learn_beta:
            self.beta = nn.Parameter(beta_tensor)
        else:
            self.register_buffer('beta', beta_tensor)
        self.threshold = threshold
        self.alpha = alpha
        self.reset_mechanism = reset_mechanism

    def forward(self,
                cur: torch.Tensor,
                mem: Optional[torch.Tensor] = None,
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        beta = self.beta.clamp(0.0, 1.0)

        if mem is None:
            mem = torch.zeros_like(cur)

        # Leaky integration
        mem = beta * mem + cur

        # Spike with ATan surrogate gradient
        spike = _atan_surrogate(mem - self.threshold, self.alpha)

        # Reset (detach so reset path carries no gradient)
        if self.reset_mechanism == 'subtract':
            mem = mem - spike.detach() * self.threshold
        else:  # 'zero'
            mem = mem * (1 - spike.detach())

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
                 reset_mechanism: str = 'subtract'):
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
