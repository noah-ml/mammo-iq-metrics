"""Tests for src/roi_metrics.py (v3.1)."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
TOOLKIT_PATH = HERE.parent / "src" / "roi_metrics.py"
spec = importlib.util.spec_from_file_location("roi_metrics", str(TOOLKIT_PATH))
mod = importlib.util.module_from_spec(spec)
sys.modules["roi_metrics"] = mod
spec.loader.exec_module(mod)
tk = mod


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

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

    def test_fixed_normalization_preserves_shape(self):
        img = np.random.default_rng(0).uniform(0, 1000, (128, 128)).astype(np.float32)
        params = tk.estimate_fixed_normalization(img)
        out = tk.apply_fixed_normalization(img, params)
        assert out.shape == img.shape
        assert out.dtype == np.float32


# ---------------------------------------------------------------------------
# Breast mask
# ---------------------------------------------------------------------------

class TestBreastMask:
    def test_mask_on_synthetic(self):
        img = np.zeros((256, 256), dtype=np.float32)
        img[50:200, 30:180] = 1000.0
        mask = tk.create_breast_mask(img)
        assert mask.any()
        assert mask.dtype == bool

    def test_mask_empty_image_returns_false(self):
        img = np.zeros((64, 64), dtype=np.float32)
        mask = tk.create_breast_mask(img)
        assert not mask.any()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

class TestMetrics:
    def test_noise_variance_constant_patch(self):
        patch = np.ones((32, 32), dtype=np.float32) * 0.5
        assert tk.compute_noise_variance_from_patch(patch) == pytest.approx(0.0)

    def test_noise_variance_noisy_patch(self):
        rng = np.random.default_rng(0)
        patch = rng.normal(0.5, 0.05, size=(32, 32)).astype(np.float32)
        var = tk.compute_noise_variance_from_patch(patch)
        assert 0.001 < var < 0.01

    def test_tenengrad_positive(self):
        rng = np.random.default_rng(3)
        img = rng.uniform(0, 1, (64, 64)).astype(np.float32)
        assert tk.compute_tenengrad(img) > 0

    def test_tenengrad_blurred_less_than_sharp(self):
        rng = np.random.default_rng(7)
        img = rng.uniform(0, 1, (64, 64)).astype(np.float32)
        blurred = tk.apply_resolution_degradation(img, scale_factor=0.33)
        assert tk.compute_tenengrad(blurred) < tk.compute_tenengrad(img)

    def test_ssim_identical(self):
        img = np.random.default_rng(1).uniform(0, 1, (64, 64)).astype(np.float32)
        assert tk.compute_ssim_fullref(img, img) == pytest.approx(1.0)

    def test_ssim_degraded_less_than_one(self):
        rng = np.random.default_rng(2)
        img = rng.uniform(0, 1, (64, 64)).astype(np.float32)
        noisy = tk.apply_dose_based_noise(img, dose_factor=0.35, baseline_noise_std=0.05, seed=0)
        assert tk.compute_ssim_fullref(img, noisy) < 1.0


# ---------------------------------------------------------------------------
# Degradations
# ---------------------------------------------------------------------------

class TestDoseBassedNoise:
    def test_full_dose_no_change(self):
        img = np.ones((32, 32), dtype=np.float32) * 0.5
        out = tk.apply_dose_based_noise(img, dose_factor=1.0, baseline_noise_std=0.01)
        np.testing.assert_array_equal(out, img)

    def test_low_dose_changes_image(self):
        img = np.ones((64, 64), dtype=np.float32) * 0.5
        out = tk.apply_dose_based_noise(img, dose_factor=0.35, baseline_noise_std=0.02, seed=42)
        assert not np.allclose(img, out)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_deterministic_with_seed(self):
        img = np.random.default_rng(0).uniform(0, 1, (64, 64)).astype(np.float32)
        a = tk.apply_dose_based_noise(img, dose_factor=0.5, baseline_noise_std=0.02, seed=7)
        b = tk.apply_dose_based_noise(img, dose_factor=0.5, baseline_noise_std=0.02, seed=7)
        np.testing.assert_array_equal(a, b)


class TestMotionBlur:
    def test_reduces_sharpness(self):
        rng = np.random.default_rng(0)
        img = rng.uniform(0, 1, (128, 128)).astype(np.float32)
        blurred, _ = tk.apply_motion_blur(img, length_px=17)
        assert tk.compute_tenengrad(blurred) < tk.compute_tenengrad(img)

    def test_preserves_shape_and_range(self):
        img = np.random.default_rng(1).uniform(0, 1, (64, 64)).astype(np.float32)
        blurred, kernel = tk.apply_motion_blur(img, length_px=7)
        assert blurred.shape == img.shape
        assert blurred.min() >= 0.0
        assert blurred.max() <= 1.0

    def test_kernel_sums_to_one(self):
        kernel = tk.build_motion_blur_kernel(9)
        assert kernel.sum() == pytest.approx(1.0, abs=1e-5)


class TestContrastReduction:
    def test_alpha_one_no_change(self):
        img = np.random.default_rng(0).uniform(0, 1, (32, 32)).astype(np.float32)
        out = tk.apply_contrast_reduction(img, alpha=1.0)
        np.testing.assert_allclose(out, img, atol=1e-6)

    def test_reduces_range(self):
        img = np.random.default_rng(1).uniform(0, 1, (64, 64)).astype(np.float32)
        out = tk.apply_contrast_reduction(img, alpha=0.5)
        assert (out.max() - out.min()) < (img.max() - img.min())

    def test_output_in_range(self):
        img = np.random.default_rng(2).uniform(0, 1, (64, 64)).astype(np.float32)
        out = tk.apply_contrast_reduction(img, alpha=0.7)
        assert out.min() >= 0.0
        assert out.max() <= 1.0


class TestResolutionDegradation:
    def test_preserves_shape(self):
        img = np.random.default_rng(1).uniform(0, 1, (128, 128)).astype(np.float32)
        out = tk.apply_resolution_degradation(img, scale_factor=0.5)
        assert out.shape == img.shape

    def test_scale_one_no_change(self):
        img = np.random.default_rng(0).uniform(0, 1, (64, 64)).astype(np.float32)
        out = tk.apply_resolution_degradation(img, scale_factor=1.0)
        np.testing.assert_array_equal(out, img)

    def test_reduces_sharpness(self):
        rng = np.random.default_rng(5)
        img = rng.uniform(0, 1, (128, 128)).astype(np.float32)
        out = tk.apply_resolution_degradation(img, scale_factor=0.33)
        assert tk.compute_tenengrad(out) < tk.compute_tenengrad(img)


# ---------------------------------------------------------------------------
# Degradation plan
# ---------------------------------------------------------------------------

class TestDegradationPlan:
    def test_plan_length(self):
        plan = tk.build_default_degradation_plan()
        assert len(plan) == 31  # 1 baseline + 6×5 degradation types

    def test_first_spec_is_baseline(self):
        plan = tk.build_default_degradation_plan()
        assert plan[0].degradation_type == "none"
        assert plan[0].severity == 0

    def test_all_types_present(self):
        plan = tk.build_default_degradation_plan()
        types = {s.degradation_type for s in plan}
        assert types == {"none", "noise", "motion_blur", "contrast", "jpeg2000", "resolution"}

    def test_severity_range(self):
        plan = tk.build_default_degradation_plan()
        for spec in plan:
            if spec.degradation_type != "none":
                assert 1 <= spec.severity <= 6

    def test_apply_degradation_dispatcher(self):
        img = np.random.default_rng(0).uniform(0, 1, (64, 64)).astype(np.float32)
        for dtype in ("noise", "motion_blur", "contrast", "resolution"):
            out = tk.apply_degradation(
                img, dtype, severity=3,
                baseline_noise_std=0.02, seed=0,
            )
            assert out.shape == img.shape
            assert out.min() >= 0.0
            assert out.max() <= 1.0
