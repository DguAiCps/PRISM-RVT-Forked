"""Spiking neuron layers for SNN backbone."""
import math
from typing import Optional, Tuple, Union

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
                 reset_mechanism: str = 'subtract',
                 channels: Optional[int] = None,
                 beta_spread: float = 0.0,
                 learn_reset: bool = False,
                 reset_ratio_init: float = 1.0,
                 reset_spread: float = 0.0):
        super().__init__()
        assert reset_mechanism in ('subtract', 'zero'), \
            f"reset_mechanism must be 'subtract' or 'zero', got '{reset_mechanism}'"
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
        self.alpha = alpha
        self.reset_mechanism = reset_mechanism

        # Learnable reset ratio: reset_amount = clamp(reset_ratio, 0, 1) * threshold
        # 1.0 = standard LIF (full reset), 0.0 = plateau (no reset)
        self.learn_reset = learn_reset
        if learn_reset:
            if channels is not None:
                reset_tensor = torch.empty(1, channels, 1, 1)
                nn.init.uniform_(reset_tensor,
                                 reset_ratio_init - reset_spread,
                                 reset_ratio_init + reset_spread)
                reset_tensor.clamp_(0.0, 1.0)
            else:
                reset_tensor = torch.tensor(reset_ratio_init)
            self.reset_ratio = nn.Parameter(reset_tensor)

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
            if self.learn_reset:
                reset_amount = self.reset_ratio.clamp(0.0, 1.0) * self.threshold
            else:
                reset_amount = self.threshold
            mem = mem - spike.detach() * reset_amount
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
                 reset_mechanism: str = 'subtract',
                 channelwise_beta: bool = False,
                 beta_spread: float = 0.0,
                 learn_reset: bool = False,
                 reset_ratio_init: float = 1.0,
                 reset_spread: float = 0.0):
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
            beta_spread=beta_spread,
            channels=out_channels if channelwise_beta else None,
            learn_reset=learn_reset,
            reset_ratio_init=reset_ratio_init,
            reset_spread=reset_spread,
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


class PlateauLIFNeuron(nn.Module):
    """Plateau Integrate-and-Fire neuron with subtractive forget gate.

    PlateauIF: no leak, no reset — membrane accumulates charge indefinitely
    and spikes persist once above threshold (plateau behavior).

    Forget gate prevents saturation via input-dependent subtractive pressure:
        gate_input = W_gate(cur) + tonic
        v_gate[t] = decay * v_gate[t-1] + gate_input   (LIF temporal integration)
        forget[t] = relu(v_gate[t])                     (non-negative)
        v_pif[t]  = v_pif[t-1] - forget[t] + cur        (subtractive forget)
        spike[t]  = H(v_pif[t] - threshold)              (no reset)

    The gate LIF converts sparse/bursty input current into sustained forget
    pressure via temporal integration. The tonic current prevents the gate
    from going completely silent.

    States returned as flat tuple: (v_pif, v_gate) for RVT framework compat.
    """

    def __init__(self,
                 channels: int,
                 threshold: float = 1.0,
                 gate_decay_init: float = 0.5,
                 gate_threshold: float = 0.5,
                 tonic_init: float = 0.1,
                 alpha: float = 2.0):
        super().__init__()
        self.threshold = threshold
        self.alpha = alpha

        # Forget gate components
        self.W_gate = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.gate_lif = LIFNeuron(
            beta_init=gate_decay_init,
            learn_beta=True,
            threshold=gate_threshold,
            alpha=alpha,
            reset_mechanism='subtract',
        )
        self.gate_tonic = nn.Parameter(torch.tensor(tonic_init))

    def forward(self,
                cur: torch.Tensor,
                mem_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            cur: (N, C, H, W) input current (output of Conv+BN)
            mem_state: (v_pif, v_gate) or None
        Returns:
            spike: (N, C, H, W) binary
            v_pif: (N, C, H, W) plateau membrane potential
            v_gate: (N, C, H, W) gate LIF membrane potential
        """
        if mem_state is None:
            v_pif = torch.zeros_like(cur)
            v_gate = None
        else:
            v_pif, v_gate = mem_state

        # Forget gate: LIF temporal integration → relu for non-negative forget
        gate_input = self.W_gate(cur) + self.gate_tonic
        _, v_gate = self.gate_lif(gate_input, v_gate)
        forget = torch.relu(v_gate)

        # Plateau IF: accumulate with subtractive forget, no reset
        v_pif = v_pif - forget + cur
        spike = _atan_surrogate(v_pif - self.threshold, self.alpha)

        return spike, v_pif, v_gate


class PlateauSpikingConvBlock(nn.Module):
    """Conv2d + BatchNorm2d + PlateauLIF neuron.

    Same interface as SpikingConvBlock but the neuron is PlateauLIF
    (no leak, no reset, with subtractive forget gate).

    State: (v_pif, v_gate) — two tensors of shape (N, C_out, H', W').
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 padding: int = 1,
                 threshold: float = 1.0,
                 gate_decay_init: float = 0.5,
                 gate_threshold: float = 0.5,
                 tonic_init: float = 0.1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.plateau_lif = PlateauLIFNeuron(
            channels=out_channels,
            threshold=threshold,
            gate_decay_init=gate_decay_init,
            gate_threshold=gate_threshold,
            tonic_init=tonic_init,
        )

    def forward(self,
                x: torch.Tensor,
                mem_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (N, C_in, H, W)
            mem_state: (v_pif, v_gate) or None
        Returns:
            spike: (N, C_out, H', W') binary
            v_pif: plateau membrane
            v_gate: gate membrane
        """
        cur = self.bn(self.conv(x))
        spike, v_pif, v_gate = self.plateau_lif(cur, mem_state)
        return spike, v_pif, v_gate
