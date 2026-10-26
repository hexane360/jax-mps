"""Tests for buffer donation (``jax.jit(donate_argnums=...)``) on MPS."""

from __future__ import annotations

import jax
import numpy as np
import pytest

# Large enough that a retained 64 MiB input is unambiguous against the noise of
# whatever else the process has allocated.
_MIB = 1024 * 1024
_BIG_BYTES = 64 * _MIB
_BIG_N = _BIG_BYTES // 4  # float32


def _on_device(array, device):
    out = jax.device_put(array, device)
    out.block_until_ready()
    return out


@pytest.mark.parametrize("donate_argnums", [0, (0, 1)])
def test_non_donatable_input_indices(mps_device, donate_argnums):
    """Passing one array to a donated parameter and to another parameter.

    Like XLA's other backends, this is rejected rather than silently skipping
    the donation; the buffer must stay usable afterwards.
    """
    fn = jax.jit(lambda u, v: u + v, donate_argnums=donate_argnums)

    s = _on_device(np.arange(4, dtype=np.float32), mps_device)

    with pytest.raises(jax.errors.JaxRuntimeError, match="previously donated"):
        fn(s, s)

    assert not s.is_deleted(), "rejected call must not consume the buffer"
    np.testing.assert_allclose(np.asarray(s), np.arange(4))


def test_donation_saves_memory(mps_device):
    """Test that buffer donation actually saves memory"""

    def chain(v):
        return (v * 2.0 + 1.0) * 0.5

    def live_bytes():
        return mps_device.memory_stats()["bytes_in_use"]

    def delta(fn):
        x = _on_device(np.ones(_BIG_N, dtype=np.float32), mps_device)
        before = live_bytes()
        # `out` and `x` must both still be referenced at the measurement, or
        # the output would be collected and its bytes freed before we look.
        out = fn(x)
        out.block_until_ready()
        grew = live_bytes() - before
        del out, x
        return grew

    donated_delta = delta(jax.jit(chain, donate_argnums=0))
    kept_delta = delta(jax.jit(chain))

    assert kept_delta >= 0.9 * _BIG_BYTES, (
        f"expected the non-donating call to retain the input "
        f"(~{_BIG_BYTES} bytes), saw {kept_delta}"
    )
    assert donated_delta < 0.5 * _BIG_BYTES, (
        f"expected donation to reclaim the input, but live memory grew by "
        f"{donated_delta} bytes (non-donating call grew by {kept_delta})"
    )
