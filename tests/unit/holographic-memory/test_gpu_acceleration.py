from __future__ import annotations

import pytest


@pytest.mark.gpu
@pytest.mark.unit
def test_gpu_platform_discovery():
    gpu = pytest.importorskip("holographic_gpu", reason="GPU backend not available")
    platforms = gpu.available_platforms()
    assert isinstance(platforms, (list, tuple))
    # The list may be empty on CI, but should be iterable

