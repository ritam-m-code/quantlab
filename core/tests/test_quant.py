import pytest
import torch

from core.quant import dequantize, fake_quantize, qrange, quantize_tensor

torch.manual_seed(0)

# GPT-2 Conv1D layout: (in_features, out_features); output channels live on dim 1.
IN, OUT = 256, 64


def conv1d_weight(in_f=IN, out_f=OUT):
    return torch.randn(in_f, out_f) * 0.02


def frac_off_grid(w, w_hat, scale):
    """Fraction of elements whose error exceeds half a step, i.e. that were clipped."""
    return ((w - w_hat).abs() > scale / 2 + 1e-7).float().mean().item()


def test_qrange():
    assert qrange(8, True) == (-127, 127)
    assert qrange(4, True) == (-7, 7)
    assert qrange(8, False) == (0, 255)
    assert qrange(4, False) == (0, 15)


def test_round_trip_on_grid_is_exact():
    step = 0.1
    ints = torch.randint(-7, 8, (32, 16)).float()
    ints[0, 0], ints[0, 1] = 7, -7  # pin absmax so the recovered scale is exactly `step`
    w = ints * step
    assert torch.allclose(fake_quantize(w, bits=4, symmetric=True), w, atol=1e-6)


def test_asymmetric_round_trip_on_grid_is_exact():
    lo, step = 0.3, 0.05
    ints = torch.randint(0, 16, (32, 16)).float()
    ints[0, 0], ints[0, 1] = 0, 15
    w = lo + ints * step
    assert torch.allclose(fake_quantize(w, bits=4, symmetric=False), w, atol=1e-6)


def test_int8_per_tensor_error_bounded_by_half_step():
    w = conv1d_weight()
    q, scale, zp = quantize_tensor(w, bits=8, symmetric=True)
    assert scale.dim() == 0 and zp.item() == 0
    err = (dequantize(q, scale, zp) - w).abs().max()
    assert err <= scale / 2 + 1e-7


def test_q_is_integer_and_in_range():
    w = conv1d_weight()
    for sym in (True, False):
        qmin, qmax = qrange(4, sym)
        q, _, _ = quantize_tensor(w, bits=4, symmetric=sym, axis=1)
        assert torch.equal(q, q.round())
        assert q.min() >= qmin and q.max() <= qmax


def test_per_channel_scale_shape_follows_conv1d_layout():
    # The GPT-2 gotcha: one scale per OUTPUT channel means reducing over dim 0.
    w = conv1d_weight()
    _, scale, _ = quantize_tensor(w, bits=8, symmetric=True, axis=1)
    assert scale.shape == (1, OUT)
    assert torch.allclose(scale.squeeze(0), w.abs().amax(dim=0) / 127)


def test_per_channel_beats_per_tensor_with_one_loud_channel():
    w = conv1d_weight()
    w[:, 5] *= 50  # one output channel with huge variance
    per_tensor = fake_quantize(w, bits=4, symmetric=True)
    per_chan = fake_quantize(w, bits=4, symmetric=True, axis=1)
    mse = lambda w_hat: ((w_hat - w) ** 2).mean()
    assert mse(per_chan) <= mse(per_tensor)


@pytest.mark.parametrize("symmetric", [True, False])
def test_group_size_equal_to_in_features_is_per_channel(symmetric):
    w = conv1d_weight()
    grouped = fake_quantize(w, bits=4, symmetric=symmetric, axis=1, group_size=IN)
    per_chan = fake_quantize(w, bits=4, symmetric=symmetric, axis=1)
    assert torch.allclose(grouped, per_chan)


def test_smaller_groups_lower_error():
    w = conv1d_weight()
    mse = lambda w_hat: ((w_hat - w) ** 2).mean()
    per_chan = fake_quantize(w, bits=4, symmetric=True, axis=1)
    g64 = fake_quantize(w, bits=4, symmetric=True, axis=1, group_size=64)
    assert mse(g64) <= mse(per_chan)


@pytest.mark.parametrize("symmetric", [True, False])
def test_clip_100_clips_nothing_and_99_clips_about_one_percent(symmetric):
    w = torch.randn(400, 250)  # 100k samples so the 1% estimate is tight
    q, scale, zp = quantize_tensor(w, bits=8, symmetric=symmetric, clip_pct=100)
    assert frac_off_grid(w, dequantize(q, scale, zp), scale) == 0.0

    q, scale, zp = quantize_tensor(w, bits=8, symmetric=symmetric, clip_pct=99)
    assert 0.005 < frac_off_grid(w, dequantize(q, scale, zp), scale) < 0.015


@pytest.mark.parametrize(
    "cfg",
    [
        dict(bits=8, symmetric=True),
        dict(bits=4, symmetric=False, axis=1),
        dict(bits=4, symmetric=True, axis=1, group_size=128),
        dict(bits=4, symmetric=False, axis=1, group_size=128, clip_pct=99.9),
    ],
)
def test_fake_quantize_preserves_shape_and_dtype(cfg):
    w = conv1d_weight()
    w_hat = fake_quantize(w, **cfg)
    assert w_hat.shape == w.shape
    assert w_hat.dtype == w.dtype
