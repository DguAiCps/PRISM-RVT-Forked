"""PhaseCodedConv2d: Convolutional phase-coded recurrent module for RVT.

Drop-in replacement for DWSConvLSTM2d / PeLIFConv2d. Channels are divided
into groups; each group has its own cycle period P_g (= n_bits_g) and
threshold ladder (optionally its own v_th_g). Within a group, every P_g
timesteps each neuron encodes its accumulated membrane into N_g bits via
FS (successive-subtraction) encoding; the encoded bits stream out one
per subsequent timestep on the recurrent path.

Multi-rate channel groups mirror ``PeLIFConv2d``'s clock-gated design
but replace the binary fire decision with a differentiable N-bit
encoding per group.

Interface matches ``PeLIFConv2d``:
``forward(x, state) -> (v_new, new_bit_buffers, step_next)`` where the
first element is the downstream feature (membrane) and the remaining
two form the per-step state. ``new_bit_buffers`` is a ``List[Tensor]``,
one tensor per group; RVT's ``recursive_detach`` / ``recursive_reset``
handle lists of tensors by recursion.

State tuple:
    v:            (N, C, H, W)               membrane potential
    bit_buffers:  List[Tensor], one per group:
                  (N, C_g, H, W, P_g)         LSB-first streaming bits
    step:         (N,) long                  timestep counter

Notes:
    * Scalar step handling (``step[0].item()``) matches PeLIFConv2d
      — RVT streams all batch elements synchronously within a call.
    * Channels are split evenly, remainder assigned to early groups
      (PeLIFNeuron2d convention).
    * W_x, W_r, dws_x are shared across all channels; group-specific
      behavior is confined to the threshold ladder and the per-phase
      encoding / streaming schedule.
"""

from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from .phase_coded_spiking import fs_stream_fn, make_thresholds


class PhaseCodedConv2d(nn.Module):
    """Convolutional phase-coded recurrent layer with channel-group multi-rate.

    Args:
        dim: number of channels (input == output)
        n_bits_groups: per-group n_bits (each also equals that group's
            period P_g). PeLIF-like default is ``(1, 2, 4, 8)``.
        v_th: either a scalar (broadcast to all groups) or a sequence
            of length ``len(n_bits_groups)`` giving the per-group v_th.
        alpha: membrane leak factor in (0, 1), shared across groups.
        threshold_mode: 'uniform' (thermometer) or 'fs' (binary).
        surrogate_k: triangle half-width for the FS surrogate.
        normalize_peaks: bound summed surrogate magnitude to 1/k.
        dws_conv_x: if True, prepend a 3x3 DWS conv to the feedforward path.
        dws_conv_kernel_size: kernel size for that DWS conv.
        cell_update_dropout: dropout applied to the streamed bit q_t.
    """

    def __init__(self,
                 dim: int,
                 n_bits_groups: Sequence[int] = (1, 2, 4, 8),
                 v_th: Union[float, Sequence[float]] = 1.0,
                 alpha: float = 0.8,
                 threshold_mode: str = 'uniform',
                 surrogate_k: float = 1.0,
                 normalize_peaks: bool = True,
                 dws_conv_x: bool = True,
                 dws_conv_kernel_size: int = 3,
                 cell_update_dropout: float = 0.0):
        super().__init__()
        self.dim = dim

        n_bits_list = [int(n) for n in n_bits_groups]
        assert len(n_bits_list) > 0, "n_bits_groups cannot be empty"
        assert all(n >= 1 for n in n_bits_list), \
            f"all n_bits must be >= 1, got {n_bits_list}"
        self.n_bits_groups = n_bits_list
        self.num_groups = len(n_bits_list)

        if isinstance(v_th, (int, float)):
            v_th_list = [float(v_th)] * self.num_groups
        else:
            v_th_list = [float(v) for v in v_th]
            assert len(v_th_list) == self.num_groups, (
                f"v_th length {len(v_th_list)} does not match "
                f"n_bits_groups length {self.num_groups}")
        self.v_th_groups = v_th_list

        self.alpha = alpha
        self.threshold_mode = threshold_mode
        self.surrogate_k = surrogate_k
        self.normalize_peaks = normalize_peaks

        # Divide channels across groups (remainder to early groups,
        # matching PeLIFNeuron2d convention).
        base = dim // self.num_groups
        rem = dim % self.num_groups
        self.group_sizes: List[int] = [
            base + (1 if i < rem else 0) for i in range(self.num_groups)
        ]
        assert all(gs > 0 for gs in self.group_sizes), (
            f"dim={dim} too small to split into {self.num_groups} nonempty "
            f"groups (group_sizes={self.group_sizes})")
        offsets = [0]
        for gs in self.group_sizes:
            offsets.append(offsets[-1] + gs)
        self.group_offsets: List[int] = offsets  # length = num_groups + 1

        # Per-group threshold buffers. Lengths differ across groups so we
        # register them individually rather than stacking.
        for g, (N, vt) in enumerate(zip(self.n_bits_groups, self.v_th_groups)):
            self.register_buffer(
                f'thresholds_{g}', make_thresholds(N, vt, threshold_mode))

        # Feedforward path: optional 3x3 DWS + 1x1 (shared across groups)
        if dws_conv_x:
            self.dws_x = nn.Conv2d(dim, dim,
                                   kernel_size=dws_conv_kernel_size,
                                   padding=dws_conv_kernel_size // 2,
                                   groups=dim)
        else:
            self.dws_x = nn.Identity()
        self.W_x = nn.Conv2d(dim, dim, kernel_size=1)

        # Recurrent path: 1x1 only, binary q_t in (sparse-compute friendly).
        self.W_r = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        with torch.no_grad():
            w = torch.empty(dim, dim)
            nn.init.orthogonal_(w, gain=1.0)
            self.W_r.weight.copy_(w.view(dim, dim, 1, 1))

        self.dropout = (nn.Dropout2d(cell_update_dropout)
                        if cell_update_dropout > 0 else None)

    def get_thresholds(self, g: int) -> torch.Tensor:
        return getattr(self, f'thresholds_{g}')

    def _init_state(self, x: torch.Tensor) -> Tuple[torch.Tensor,
                                                    List[torch.Tensor],
                                                    torch.Tensor]:
        N, _, H, W = x.shape
        v = torch.zeros_like(x)
        bit_buffers = [
            torch.zeros(N, gs, H, W, Pg, dtype=x.dtype, device=x.device)
            for gs, Pg in zip(self.group_sizes, self.n_bits_groups)
        ]
        step = torch.zeros(N, dtype=torch.long, device=x.device)
        return v, bit_buffers, step

    def forward(self,
                x: torch.Tensor,
                h_and_c_previous: Optional[Tuple[torch.Tensor,
                                                 List[torch.Tensor],
                                                 torch.Tensor]] = None
                ) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
        """One timestep forward.

        Args:
            x: (N, C, H, W) dense input feature from upstream.
            h_and_c_previous: None or (v, bit_buffers, step) where
                bit_buffers is a list of per-group tensors.

        Returns:
            (v_new, new_bit_buffers, step_next).
            v_new is the downstream feature; new_bit_buffers is a list
            aligned with self.n_bits_groups.
        """
        if h_and_c_previous is None:
            v, bit_buffers, step = self._init_state(x)
        else:
            v, bit_buffers, step = h_and_c_previous

        # Scalar phase — all batch elements share the same step under
        # RVT streaming (matches PeLIFConv2d's convention).
        t_scalar = int(step[0].item())

        # Assemble q_t by reading each group's buffer at its own phase.
        q_t_parts: List[torch.Tensor] = []
        for Pg, buf in zip(self.n_bits_groups, bit_buffers):
            phi_g = t_scalar % Pg
            q_t_parts.append(buf[..., phi_g])
        q_t = torch.cat(q_t_parts, dim=1)  # (N, C, H, W)

        q_in = (self.dropout(q_t)
                if (self.dropout is not None and self.training) else q_t)

        # State update — W_x, W_r apply to the full channel stack.
        v_minus = self.alpha * v + self.W_x(self.dws_x(x)) + self.W_r(q_in)

        # Per-group encoding / accumulation
        v_new_parts: List[torch.Tensor] = []
        new_bit_buffers: List[torch.Tensor] = []
        for g, (Pg, buf) in enumerate(zip(self.n_bits_groups, bit_buffers)):
            off = self.group_offsets[g]
            end = self.group_offsets[g + 1]
            v_minus_g = v_minus[:, off:end]
            phi_g = t_scalar % Pg

            if phi_g != Pg - 1:
                v_new_parts.append(v_minus_g)
                new_bit_buffers.append(buf)
            else:
                thresholds_g = self.get_thresholds(g)
                bits = fs_stream_fn(v_minus_g, thresholds_g,
                                    self.surrogate_k, self.normalize_peaks)
                subtract = (bits * thresholds_g).sum(dim=-1)
                v_new_parts.append(v_minus_g - subtract)
                # LSB-first streaming: bits[..., N-1-k] consumed at phase k.
                new_bit_buffers.append(bits.flip(-1))

        v_new = torch.cat(v_new_parts, dim=1)
        step_next = step + 1
        return v_new, new_bit_buffers, step_next
