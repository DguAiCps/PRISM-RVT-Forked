"""PeLIF (Periodic LIF) spiking layers for 2D backbone.

Channel-wise period groups: channels are divided into groups with different
firing periods. All channels integrate input every timestep, but only fire
at their designated clock step.
"""
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


def _atan_surrogate(mem_shift: torch.Tensor, alpha: float = 2.0) -> torch.Tensor:
    """Spike function with ATan surrogate gradient."""
    smooth = (1.0 / math.pi) * torch.atan(
        (math.pi * alpha / 2.0) * mem_shift
    ) + 0.5
    binary = (mem_shift > 0).float()
    return smooth + (binary - smooth).detach()


class PeLIFNeuron2d(nn.Module):
    """Periodic LIF neuron for 2D feature maps.

    Channels are divided into groups with different firing periods.
    All channels integrate every timestep (leaky integration),
    but only fire at their clock step (t % P == 0).

    Args:
        channels: total number of channels
        periods: list of firing periods (e.g. [1, 2, 4, 8])
        beta_init: membrane decay factor
        learn_beta: whether beta is learnable
        threshold: spike threshold
        alpha: surrogate gradient sharpness
    """

    def __init__(self,
                 channels: int,
                 periods: Tuple[int, ...] = (1, 2, 4, 8),
                 beta_init: float = 0.9,
                 learn_beta: bool = True,
                 threshold: float = 1.0,
                 alpha: float = 2.0):
        super().__init__()
        self.channels = channels
        self.periods = list(periods)
        self.num_groups = len(self.periods)
        self.threshold = threshold
        self.alpha = alpha

        # Divide channels evenly among period groups
        base = channels // self.num_groups
        remainder = channels % self.num_groups
        self.group_sizes = [base + (1 if i < remainder else 0)
                            for i in range(self.num_groups)]

        # Group boundaries (channel indices)
        self.group_offsets = [0]
        for gs in self.group_sizes:
            self.group_offsets.append(self.group_offsets[-1] + gs)

        # Per-channel beta (learnable)
        beta_tensor = torch.empty(1, channels, 1, 1)
        nn.init.uniform_(beta_tensor, beta_init - 0.05, beta_init + 0.05)
        beta_tensor.clamp_(0.0, 1.0)
        if learn_beta:
            self.beta = nn.Parameter(beta_tensor)
        else:
            self.register_buffer('beta', beta_tensor)

    def forward(self,
                cur: torch.Tensor,
                mem: Optional[torch.Tensor] = None,
                t: int = 0,
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            cur: input current (N, C, H, W)
            mem: previous membrane potential (N, C, H, W) or None
            t: current timestep (for clock gating)
        Returns:
            spike: (N, C, H, W) binary (0 for non-clock channels)
            mem: (N, C, H, W) updated membrane potential
        """
        beta = self.beta.clamp(0.0, 1.0)

        if mem is None:
            mem = torch.zeros_like(cur)

        # All channels integrate every step
        mem = beta * mem + cur

        # Per-group clock-gated firing
        spike_parts = []
        for g in range(self.num_groups):
            period = self.periods[g]
            off = self.group_offsets[g]
            end = self.group_offsets[g + 1]

            if t % period == 0:
                s_g = _atan_surrogate(mem[:, off:end] - self.threshold, self.alpha)
                # Soft reset (not detached, matching 1D PeLIF)
                mem = mem.clone()
                mem[:, off:end] = mem[:, off:end] - s_g * self.threshold
                spike_parts.append(s_g)
            else:
                spike_parts.append(torch.zeros_like(mem[:, off:end]))

        spike = torch.cat(spike_parts, dim=1)
        return spike, mem
