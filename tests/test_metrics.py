"""Basic tests for roi_metrics module."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

# Load the toolkit module
HERE = Path(__file__).resolve().parent
TOOLKIT_PATH = HERE.parent / "src" / "roi_metrics.py"
spec = importlib.util.spec_from_file_location("roi_metrics", str(TOOLKIT_PATH))
mod = importlib.util.module_from_spec(spec)
sys.modules["roi_metrics"] = mod
spec.loader.exec_module(mod)
tk = mod


class TestNormalization:
    def test_normalize_for_mask_basic(self):
        img = np.array([[0.0, 50.0], [100.0, 200.0]], dtype=np.float32)
        out = tk.normalize_for_mask(img)
        assert out.min() == pytest.approx(0.0)
        assert out.max() == pytest.approx(1.0)

    def test_fixed_normalization_roundtrip(self):
        rng = np.random.default_rng(42)
        img = rng.uniform(100, 4000, size=(64, 64)).astype(np.float32)
        params = tk.estimate_fixed_normalization(img)
        normed = tk.apply_fixed_normalization(img, params)
        assert normed.min() >= 0.0
        assert normed.max() <= 1.0


class TestBreastMask:
    def test_mask_on_synthetic(self):
        img = np.zeros((256, 256), dtype=np.float32)
        img[50:200, 30:180] = 1000.0  # synthetic breast region
        mask = tk.create_breast_mask(img)
        assert mask.any()
        assert mask.dtype == bool


class TestMetrics:
    def test_snr_constant_patch(self):
        patch = np.ones((32, 32), dtype=np.float32) * 0.5
        snr = tk.compute_snr_from_patch(patch)
        assert snr == float("inf")

    def test_snr_noisy_patch(self):
        rng = np.random.default_rng(0)
        patch = rng.normal(0.5, 0.1, size=(32, 32)).astype(np.float32)
        snr = tk.compute_snr_from_patch(patch)
        assert 3.0 < snr < 8.0

    def test_noise_variance(self):
        patch = np.ones((32, 32), dtype=np.float32) * 0.5
        assert tk.compute_noise_variance_from_patch(patch) == pytest.approx(0.0)

    def test_ssim_identical(self):
        img = np.random.default_rng(1).uniform(0, 1, (64, 64)).astype(np.float32)
        assert tk.compute_ssim_fullref(img, img) == pytest.approx(1.0)

    def test_vol_positive(self):
        rng = np.random.default_rng(2)
        img = rng.uniform(0, 1, (64, 64)).astype(np.float32)
        vol = tk.compute_variance_of_laplacian(img)
        assert vol > 0

    def test_tenengrad_positive(self):
        rng = np.random.default_rng(3)
        img = rng.uniform(0, 1, (64, 64)).astype(np.float32)
        assert tk.compute_tenengrad(img) > 0


class TestDegradations:
    def test_gaussian_noise_changes_image(self):
        img = np.ones((64, 64), dtype=np.float32) * 0.5
        rng = np.random.default_rng(0)
        noisy = tk.apply_gaussian_noise(img, sigma=0.1, rng=rng)
        assert not np.allclose(img, noisy)
        assert noisy.min() >= 0.0
        assert noisy.max() <= 1.0

    def test_gaussian_blur_reduces_sharpness(self):
        rng = np.random.default_rng(0)
        img = rng.uniform(0, 1, (64, 64)).astype(np.float32)
        blurred = tk.apply_gaussian_blur(img, sigma=2.0)
        vol_orig = tk.compute_variance_of_laplacian(img)
        vol_blur = tk.compute_variance_of_laplacian(blurred)
        assert vol_blur < vol_orig

    def test_resolution_loss_and_restore_shape(self):
        img = np.random.default_rng(1).uniform(0, 1, (128, 128)).astype(np.float32)
        out = tk.apply_resolution_loss(img, scale=0.5)
        assert out.shape == img.shape

    def test_build_default_plan(self):
        plan = tk.build_default_degradation_plan()
        assert len(plan) == 13  # 1 baseline + 4 types × 3 severities
        assert plan[0].degradation_type == "none"
