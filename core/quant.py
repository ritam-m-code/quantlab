"""Weight quantization math. Pure torch, no model code in here.

Conventions used throughout:

* ``axis`` is the *channel* axis: you get one scale per index along it, and the
  min/max (or absmax) is reduced over every other dim. For GPT-2's Conv1D the
  weight is (in_features, out_features), so per-output-channel means ``axis=1``
  (which reduces over dim 0). For nn.Linear it would be ``axis=0``.
* ``q`` is returned as a float tensor holding integer values. Real kernels pack
  these into int8/int4 storage; for teaching, floats are easier to inspect.
* ``scale`` and ``zero_point`` are always returned *broadcastable to w*, so
  ``dequantize`` never needs to know which granularity was used.
"""

from numpy import integer
import torch


def qrange(bits: int, symmetric: bool) -> tuple[int, int]:
    #asymetric:
    if not symmetric:
        qmin = 0
        qmax = 2**bits - 1
        return qmin, qmax
    #symetric:
    qmax = 2**(bits-1)-1
    return -qmax, qmax

def quantize_tensor(
    w: torch.Tensor,
    bits: int,
    symmetric: bool,
    axis: int | None = None,
    group_size: int | None = None,
    clip_pct: float = 100,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    #Quantize a weight tensor to integers, returning (q, scale, zero_point).
    qmin, qmax = qrange(bits, symmetric)

    # Arrange w as (n_groups, group_len, n_channels) so every case reduces over dim 1.
    if axis is None:
        view = w.reshape(1, -1, 1)              # per-tensor: one group, one channel
    else:
        if axis == 0:
            wt = w.T
        else:
            wt = w
        # put the channels on dim 1
        n_in, n_out = wt.shape
        g = group_size or n_in                  # per-channel = one group spanning the channel
        view = wt.reshape(n_in // g, g, n_out)  # view[k, :, j] = chunk k of channel j

    if symmetric:
        absmax = torch.quantile(view.abs(), clip_pct / 100, dim=1, keepdim=True)
        scale = (absmax / qmax).clamp_min(1e-12)
        zero_point = torch.zeros_like(scale)
    else:
        lo = torch.quantile(view, (100 - clip_pct) / 200, dim=1, keepdim=True)
        hi = torch.quantile(view, (100 + clip_pct) / 200, dim=1, keepdim=True)
        scale = ((hi - lo) / (qmax - qmin)).clamp_min(1e-12)
        zero_point = torch.round(qmin - lo / scale)

    q = torch.clamp(torch.round(view / scale) + zero_point, qmin, qmax)

    # Undo the arrangement so q matches w and scale/zero_point broadcast against it.
    if axis is None:
        return q.reshape(w.shape), scale.reshape(()), zero_point.reshape(())
    if group_size is None:
        scale, zero_point = scale.reshape(1, n_out), zero_point.reshape(1, n_out)
    else:
        scale = scale.expand_as(view).reshape(n_in, n_out)
        zero_point = zero_point.expand_as(view).reshape(n_in, n_out)
    q = q.reshape(n_in, n_out)
    if axis == 0:
        q, scale, zero_point = q.T, scale.T, zero_point.T
    return q, scale, zero_point


def dequantize(q: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor) -> torch.Tensor:
    """Map integers back to real values: ``w_hat = (q - zero_point) * scale``."""
    raise NotImplementedError


def fake_quantize(w: torch.Tensor, **cfg) -> torch.Tensor:
    """Quantize then dequantize. Same shape and dtype as w.

    This is what the sweep swaps into the model: the network still runs in fp32,
    but every weight now sits on the quantization grid, so the error is real.
    ``cfg`` is passed straight through to ``quantize_tensor``.
    """
    raise NotImplementedError
