"""Phase-coded FS encoding primitives for the RVT backbone.

Mirrors the multi-peak triangle surrogate used in the sMNIST BitPlane FS
recurrent module (see the parent PRISM-Module-Test repo,
``src/surrogates/surrogate_spike.py``). Kept self-contained here so the
RVT submodule does not depend on the parent repo.

Forward: successive subtraction against N descending thresholds
    r_0 = v;   b_k = 1[r_k >= theta_k];   r_{k+1} = r_k - b_k * theta_k
Returns only the bits tensor (shape = v.shape + (N,)). The caller
recomputes the residual as ``v - (bits * thresholds).sum(-1)`` so that
autograd tracks the residual path through the bits surrogate and the
linear subtraction in a single unified graph.

Backward: data-dependent multi-peak triangle surrogate. Each bit's peak
is centered at the residual where its comparison was actually taken.
Optional peak normalization bounds the summed peak magnitude to 1/k.
"""

import torch


class FSStreamSurrogate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v, thresholds, k, normalize_peaks):
        N = thresholds.shape[0]
        bits = torch.zeros(*v.shape, N, dtype=v.dtype, device=v.device)
        residuals = torch.zeros(*v.shape, N, dtype=v.dtype, device=v.device)
        r = v
        for i in range(N):
            residuals[..., i] = r
            b = (r >= thresholds[i]).to(v.dtype)
            bits[..., i] = b
            r = r - b * thresholds[i]
        ctx.save_for_backward(residuals, thresholds)
        ctx.k = k
        ctx.normalize_peaks = normalize_peaks
        return bits

    @staticmethod
    def backward(ctx, grad_bits):
        residuals, thresholds = ctx.saved_tensors
        k = ctx.k
        diff = residuals - thresholds
        peaks = (1.0 / k) * (1 - diff.abs() / k).clamp(min=0)
        if ctx.normalize_peaks:
            n_active = (peaks > 0).to(peaks.dtype).sum(dim=-1, keepdim=True).clamp(min=1.0)
            peaks = peaks / n_active
        grad_v = (grad_bits * peaks).sum(dim=-1)
        return grad_v, None, None, None


fs_stream_fn = FSStreamSurrogate.apply


def make_thresholds(n_bits: int, v_th: float, mode: str) -> torch.Tensor:
    """Build threshold vector of length n_bits.

    mode == 'uniform': theta_k = v_th for all k (thermometer encoding,
        encoding range [0, N*v_th] with N+1 levels spaced by v_th).
    mode == 'fs':      theta_k = v_th / 2^{k+1}, descending
        (binary encoding, range [0, v_th) with 2^N levels).
    """
    if mode == 'uniform':
        return torch.full((n_bits,), v_th, dtype=torch.float32)
    if mode == 'fs':
        return torch.tensor(
            [v_th / (2.0 ** (k + 1)) for k in range(n_bits)],
            dtype=torch.float32,
        )
    raise ValueError(f"unknown threshold mode: {mode}")
