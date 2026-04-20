"""PhaseCodedConv2d: Convolutional phase-coded recurrent module for RVT.

Drop-in replacement for DWSConvLSTM2d / PeLIFConv2d. Every P timesteps
each (n, c, h, w) neuron encodes its accumulated membrane potential into
N bits via FS (successive-subtraction) encoding; the encoded bits are
streamed out one per subsequent timestep on the recurrent path.

Adapted from the 1D BitPlane FS recurrent module in the parent repo
(``src/layers/bitplane_fs_recurrent.py``). Interface matches
``PeLIFConv2d``: ``forward(x, state) -> (mem, new_buffer, next_step)``.
The returned ``mem`` is the membrane potential ``v`` and is used by
downstream modules (FPN, detection head).

State tuple (3 tensors so RVT's ``recursive_detach`` / ``recursive_reset``
handle them uniformly):
    v:          (N, C, H, W) membrane potential, also the downstream feature
    bit_buffer: (N, C, H, W, P) pre-encoded bits, indexed by phase
    step:       (N,) long tensor timestep counter

Notes:
    * Scalar step handling (``step[0].item()``) matches the PeLIFConv2d
      convention — RVT streams all batch elements synchronously within a
      forward call.
    * The recurrent path ``W_r`` is 1x1 only; together with the binary
      streaming bit ``q_t``, this keeps the downstream-to-recurrent path
      addition-only / sparse-compute friendly.
    * LSB-first streaming: ``bit_buffer[..., k]`` is consumed at phase
      ``k`` of the next cycle, so we store ``bits.flip(-1)``.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from .phase_coded_spiking import fs_stream_fn, make_thresholds


class PhaseCodedConv2d(nn.Module):
    """Convolutional phase-coded recurrent layer.

    Args:
        dim: number of channels (input == output)
        n_bits: bits per encoding cycle (= period P)
        v_th: spike threshold (scales the threshold ladder)
        alpha: membrane leak factor in (0, 1)
        threshold_mode: 'uniform' (thermometer) or 'fs' (binary)
        surrogate_k: triangle half-width for the FS surrogate
        normalize_peaks: bound the summed surrogate magnitude to 1/k
        dws_conv_x: if True, prepend a 3x3 DWS conv to the feedforward path
        dws_conv_kernel_size: kernel size for that DWS conv
        cell_update_dropout: dropout applied to the streamed bit q_t
    """

    def __init__(self,
                 dim: int,
                 n_bits: int = 4,
                 v_th: float = 1.0,
                 alpha: float = 0.8,
                 threshold_mode: str = 'uniform',
                 surrogate_k: float = 1.0,
                 normalize_peaks: bool = True,
                 dws_conv_x: bool = True,
                 dws_conv_kernel_size: int = 3,
                 cell_update_dropout: float = 0.0):
        super().__init__()
        self.dim = dim
        self.n_bits = n_bits
        self.period = n_bits  # P = N (one bit emitted per step across the cycle)
        self.alpha = alpha
        self.v_th = v_th
        self.threshold_mode = threshold_mode
        self.surrogate_k = surrogate_k
        self.normalize_peaks = normalize_peaks

        thresholds = make_thresholds(n_bits, v_th, threshold_mode)
        self.register_buffer('thresholds', thresholds)

        # Feedforward path: optional 3x3 DWS + 1x1
        if dws_conv_x:
            self.dws_x = nn.Conv2d(dim, dim,
                                   kernel_size=dws_conv_kernel_size,
                                   padding=dws_conv_kernel_size // 2,
                                   groups=dim)
        else:
            self.dws_x = nn.Identity()
        self.W_x = nn.Conv2d(dim, dim, kernel_size=1)

        # Recurrent path: 1x1 only (binary q_t in, sparse-compute friendly)
        self.W_r = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        with torch.no_grad():
            w = torch.empty(dim, dim)
            nn.init.orthogonal_(w, gain=1.0)
            self.W_r.weight.copy_(w.view(dim, dim, 1, 1))

        self.dropout = nn.Dropout2d(cell_update_dropout) if cell_update_dropout > 0 else None

    def forward(self,
                x: torch.Tensor,
                h_and_c_previous: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One timestep forward.

        Args:
            x: (N, C, H, W) dense input feature from upstream
            h_and_c_previous: None or (v, bit_buffer, step)

        Returns:
            (v_new, new_bit_buffer, step_next).
            ``v_new`` is the downstream feature (h_c_tuple[0] convention).
        """
        N, C, H, W = x.shape
        P = self.period

        if h_and_c_previous is None:
            v = torch.zeros_like(x)
            bit_buffer = torch.zeros(N, C, H, W, P, dtype=x.dtype, device=x.device)
            step = torch.zeros(N, dtype=torch.long, device=x.device)
        else:
            v, bit_buffer, step = h_and_c_previous

        # Scalar phase — all batch elements share the same step under RVT
        # streaming (see PeLIFConv2d for the same convention).
        t_scalar = int(step[0].item())
        phi = t_scalar % P

        # q_t: bit to emit this step, read from the phase-phi slot.
        q_t = bit_buffer[..., phi]  # (N, C, H, W)
        q_in = self.dropout(q_t) if (self.dropout is not None and self.training) else q_t

        # State update (always)
        v_minus = self.alpha * v + self.W_x(self.dws_x(x)) + self.W_r(q_in)

        if phi != P - 1:
            # Accumulation: no firing, no buffer update
            v_new = v_minus
            new_bit_buffer = bit_buffer
        else:
            # Encoding onset: FS-encode v_minus pointwise over channels/space.
            # bits: v_minus.shape + (n_bits,) == (N, C, H, W, n_bits)
            bits = fs_stream_fn(v_minus, self.thresholds,
                                self.surrogate_k, self.normalize_peaks)
            # Residual via linear subtraction. Autograd follows the unified
            # graph (bits surrogate + subtraction), no separate STE path.
            subtract = (bits * self.thresholds).sum(dim=-1)  # (N, C, H, W)
            v_new = v_minus - subtract
            # Fill buffer with REVERSED bits for LSB-first streaming.
            # bit_buffer[..., k] is consumed at phase k of the next cycle,
            # so bit_buffer[..., k] = bits[..., N-1-k].
            new_bit_buffer = bits.flip(-1)

        step_next = step + 1
        return v_new, new_bit_buffer, step_next
