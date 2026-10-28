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
    """Quantize ``w`` to ``bits``-bit integers.

    Granularity is chosen by the arguments:

    =================  ==========================  =====================================
    per-tensor         ``axis=None``               one scale for all of w
    per-channel        ``axis=k``                  one scale per index along dim k
    group-wise         ``axis=k, group_size=g``    the *other* dim is cut into blocks
                                                   of g; one scale per (block, channel)
    =================  ==========================  =====================================

    Group-wise only needs to support 2-D w. Its reduction dim must be divisible
    by ``group_size``. Hint: for a (in, out) weight with axis=1, view it as
    ``(in // g, g, out)`` and reduce over dim 1 with ``keepdim=True``.

    Steps:

    1. Calibration range per scale-group (the elements that share one scale):
       symmetric  -> ``absmax``. With clipping, use the ``clip_pct`` percentile of
                     ``|w|`` instead of the true max.
       asymmetric -> ``(lo, hi)``. With clipping, use the ``(100 - clip_pct) / 2`` and
                     ``(100 + clip_pct) / 2`` percentiles, so the clipped mass is
                     split between both tails. ``clip_pct=100`` must equal the true
                     min/max. (``torch.quantile`` takes q in [0, 1] and a ``dim``.)
    2. Scale and zero point:
       symmetric  -> ``scale = absmax / qmax``, ``zero_point = 0``
       asymmetric -> ``scale = (hi - lo) / (qmax - qmin)``,
                     ``zero_point = round(qmin - lo / scale)``
       Guard against ``scale == 0`` (a constant channel), e.g. ``clamp_min(1e-12)``.
    3. ``q = clamp(round(w / scale) + zero_point, qmin, qmax)``

    Returns:
        q: same shape as w, integer-valued floats.
        scale, zero_point: broadcastable to w. For per-tensor these are 0-d;
            for per-channel with axis=1 on 2-D w, shape (1, out); for group-wise,
            expand back to w's full shape (each block's scale repeated).
    """
    raise NotImplementedError


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
