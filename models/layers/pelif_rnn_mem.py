"""PeLIFConv2d: Convolutional PeLIF recurrent module for the RVT backbone.

Drop-in replacement for DWSConvLSTM2d. Instead of LSTM gates, uses PeLIF
dynamics (LIF with period-based clock gating) to achieve high execution
sparsity on the recurrent path.

Design:
    I[t] = W_x(x[t]) + W_rec(s[t-1])
    (spike, mem) = PeLIFNeuron2d(I[t], mem_prev, t=step)

Reuses PeLIFNeuron2d from pelif_spiking.py for the neuron dynamics.

Feedforward path (W_x):
    optional 3x3 depthwise-separable conv (default on) followed by 1x1 conv.
    x is a dense feature map from the backbone, so DWS does not affect
    sparsity; it only adds a spatial receptive field to the input current.

Recurrent path (W_rec):
    1x1 conv only. s[t-1] is a sparse binary spike tensor, so restricting
    the recurrent path to 1x1 preserves the sparse-compute potential
    (the SOP savings that motivate this module). Period clock gating
    zeros out entire channel groups at non-clock steps, which further
    reduces the non-zero support of s[t-1].

State tuple (flat 3-element, compatible with RVT state management):
    (mem, prev_spike, step)
    - mem: (N, C, H, W) membrane potential — also the downstream feature
    - prev_spike: (N, C, H, W) last spike output — used as W_rec input next step
    - step: (N,) long tensor timestep counter (one per batch element)

    All three are tensors so RVT's `recursive_detach` / `recursive_reset`
    (in modules/utils/detection.py) handle them correctly. The caller uses
    `h_c_tuple[0]` to read the output feature, matching DWSConvLSTM2d's
    (h, c) convention where element 0 is the feature.

Downstream output semantics:
    Unlike the spike `s`, which is binary and sparse (no stable value without
    events), the membrane `u` is a dense float that persists across timesteps
    via `alpha * u_prev + I`. Downstream modules (next MaxViT stage, FPN, YOLO
    head) receive `u` so they see a stable feature like the hidden state of
    an LSTM. The recurrent path *inside* this layer still uses spikes to keep
    W_rec sparse (the core SOP saving), so sparsity story is preserved on the
    W_rec weights even though the outward-facing output is dense.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from .pelif_spiking import PeLIFNeuron2d


class PeLIFConv2d(nn.Module):
    """Convolutional PeLIF recurrent layer.

    Args:
        dim: number of channels (input == output)
        periods: clock periods for channel groups (e.g. (1, 2, 4, 8))
        v_th: spike threshold
        beta_init: initial membrane decay factor
        learn_beta: whether beta is learnable per channel
        dws_conv_x: if True, apply 3x3 DWS conv before 1x1 on feedforward path
        dws_conv_kernel_size: kernel size for the feedforward DWS conv
        surrogate: surrogate gradient type ('triangle' or 'atan')
        cell_update_dropout: dropout on the recurrent spike input
    """

    def __init__(self,
                 dim: int,
                 periods: Tuple[int, ...] = (1, 2, 4, 8),
                 v_th: float = 1.0,
                 beta_init: float = 0.9,
                 learn_beta: bool = True,
                 dws_conv_x: bool = True,
                 dws_conv_kernel_size: int = 3,
                 surrogate: str = 'triangle',
                 cell_update_dropout: float = 0.0):
        super().__init__()
        self.dim = dim

        # Feedforward path: optional 3x3 DWS + 1x1
        if dws_conv_x:
            self.dws_x = nn.Conv2d(dim, dim,
                                   kernel_size=dws_conv_kernel_size,
                                   padding=dws_conv_kernel_size // 2,
                                   groups=dim)
        else:
            self.dws_x = nn.Identity()
        self.W_x = nn.Conv2d(dim, dim, kernel_size=1)

        # Recurrent path: 1x1 only (preserve sparsity from sparse s[t-1])
        self.W_rec = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        with torch.no_grad():
            w = torch.empty(dim, dim)
            nn.init.orthogonal_(w, gain=1.0)
            self.W_rec.weight.copy_(w.view(dim, dim, 1, 1))

        # PeLIF neuron dynamics (reused from pelif_spiking.py)
        self.neuron = PeLIFNeuron2d(
            channels=dim,
            periods=periods,
            beta_init=beta_init,
            learn_beta=learn_beta,
            threshold=v_th,
            surrogate=surrogate,
        )

        self.dropout = nn.Dropout2d(cell_update_dropout) if cell_update_dropout > 0 else None

    def forward(self,
                x: torch.Tensor,
                h_and_c_previous: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (N, C, H, W) dense input feature from backbone
            h_and_c_previous: None or (mem, prev_spike, step)

        Returns:
            (mem, spike, step_next) — flat 3-element tuple.
            mem is the feature used by downstream modules (h_c_tuple[0]),
            providing a dense float signal that persists across timesteps.
            spike is kept as part of the state to feed W_rec on the next call,
            preserving the sparse recurrent path.
        """
        N = x.shape[0]
        if h_and_c_previous is None:
            mem = torch.zeros_like(x)
            s_prev = torch.zeros_like(x)
            step = torch.zeros(N, dtype=torch.long, device=x.device)
        else:
            mem, s_prev, step = h_and_c_previous
            # Batch elements whose step was zeroed by RVT's recursive_reset
            # (new sequence) also have zeroed mem/s_prev from the same call.

        s_in = self.dropout(s_prev) if self.dropout is not None and self.training else s_prev

        cur = self.W_x(self.dws_x(x)) + self.W_rec(s_in)

        # All batch elements share the same step within a forward call under
        # RVT's streaming pipeline. Use the first element for clock gating.
        t_scalar = int(step[0].item())
        spike, mem = self.neuron(cur, mem, t=t_scalar)

        step_next = step + 1
        return mem, spike, step_next
