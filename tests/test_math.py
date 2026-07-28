"""Math sanity tests for SPEED spectral utilities.

Based on https://github.com/howardhx/speed
Run from the repo root with:
  python -m unittest tests.test_math -v
"""
import importlib.util as _iu
import pathlib
import sys
import unittest

import numpy as np
import torch

_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

_spec = _iu.spec_from_file_location("spectral_utils", str(_ROOT / "spectral_utils.py"))
_spectral = _iu.module_from_spec(_spec)
_spec.loader.exec_module(_spectral)


def _load_speed_core():
    """Load ``speed_core`` (which uses relative imports) under a synthetic package."""
    import types
    pkg_name = "_speed_test_pkg"
    pkg = types.ModuleType(pkg_name)
    pkg.__path__ = [str(_ROOT)]
    sys.modules[pkg_name] = pkg
    sys.modules[f"{pkg_name}.spectral_utils"] = _spectral
    spec = _iu.spec_from_file_location(f"{pkg_name}.speed_core", str(_ROOT / "speed_core.py"))
    module = _iu.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_speed_core = _load_speed_core()

power_spectrum = _spectral.power_spectrum
activation_time = _spectral.activation_time
delta_optimal_transitions = _spectral.delta_optimal_transitions
align_timestep = _spectral.align_timestep
kappa = _spectral.kappa
_dct_expand_np = _spectral._dct_expand_np
_dwt_expand_np = _spectral._dwt_expand_np
_fft_expand_np = _spectral._fft_expand_np
validate_scales = _spectral.validate_scales


class TestPowerSpectrum(unittest.TestCase):
    def test_shape(self):
        A, beta = 203.6, 1.92
        result = power_spectrum(10.0, A, beta)
        expected = A * (10.0 ** (-beta))
        self.assertAlmostEqual(result, expected, places=4)

    def test_basic_values(self):
        A, beta = 100.0, 2.0
        self.assertAlmostEqual(power_spectrum(1.0, A, beta), 100.0)
        self.assertAlmostEqual(power_spectrum(2.0, A, beta), 25.0)


class TestActivationTime(unittest.TestCase):
    def test_output_range(self):
        t = activation_time(P_omega=1.0, delta=0.01)
        self.assertTrue(0 < t < 1, f"expected activation time in (0,1), got {t}")

    def test_delta_rejects_ge_1(self):
        with self.assertRaises(ValueError):
            activation_time(1.0, delta=1.0)
        with self.assertRaises(ValueError):
            activation_time(1.0, delta=2.0)

    def test_smaller_delta_delays(self):
        t_big = activation_time(1.0, delta=0.1)
        t_small = activation_time(1.0, delta=0.001)
        self.assertGreater(t_small, t_big,
                           msg="smaller delta gives larger t (activates later)")


class TestDeltaOptimalTransitions(unittest.TestCase):
    def test_num_transitions(self):
        scales = [0.25, 0.5, 1.0]
        t_stars = delta_optimal_transitions(scales, delta=0.01, A=203.6, beta=1.92,
                                            H=64, W=64)
        self.assertEqual(len(t_stars), 2)

    def test_decreasing(self):
        scales = [0.25, 0.5, 1.0]
        t_stars = delta_optimal_transitions(scales, delta=0.01, A=203.6, beta=1.92,
                                            H=64, W=64)
        self.assertTrue(all(t_stars[i] >= t_stars[i + 1] for i in range(len(t_stars) - 1)),
                        msg="transition times should be non-increasing")

    def test_single_scale_no_transitions(self):
        scales = [1.0]
        t_stars = delta_optimal_transitions(scales, delta=0.01, A=203.6, beta=1.92,
                                            H=64, W=64)
        self.assertEqual(t_stars, [])


class TestPresets(unittest.TestCase):
    """Each preset's documented transition sigma must survive edits to (A, beta).

    The reference case is the one the presets are calibrated on: 1024x1024
    (128x128 latent, so omega_max = 64), ``scales=0.5,1.0`` and ``delta=0.01``.
    """

    REFERENCE_SIGMA = {
        "flux": 0.8527,
        "wan21": 0.6941,
        "krea-2": 0.9000,
        "krea-2-raw": 0.8300,
        "z-image": 0.9400,
        "flux2": 0.9000,
    }

    def _sigma_star(self, preset, scale=0.5, delta=0.01, latent=128):
        return delta_optimal_transitions(
            [scale, 1.0], delta, preset["A"], preset["beta"], latent, latent)[0]

    def test_documented_transition_sigmas(self):
        for name, expected in self.REFERENCE_SIGMA.items():
            with self.subTest(preset=name):
                self.assertAlmostEqual(
                    self._sigma_star(_speed_core._PRESETS[name]), expected, places=4)

    def test_custom_preset_is_a_sentinel(self):
        self.assertIsNone(_speed_core._PRESETS["custom"])

    def test_krea_and_zimage_keep_their_vae_family_slope(self):
        # beta comes from the measured fit for the same latent space; only the
        # amplitude is calibrated. See the preset table in speed_core.py.
        self.assertEqual(_speed_core._PRESETS["krea-2"]["beta"],
                         _speed_core._PRESETS["wan21"]["beta"])
        self.assertEqual(_speed_core._PRESETS["krea-2-raw"]["beta"],
                         _speed_core._PRESETS["wan21"]["beta"])
        self.assertEqual(_speed_core._PRESETS["z-image"]["beta"],
                         _speed_core._PRESETS["flux"]["beta"])

    def test_raw_preset_transitions_later_than_the_few_step_one(self):
        self.assertLess(self._sigma_star(_speed_core._PRESETS["krea-2-raw"]),
                        self._sigma_star(_speed_core._PRESETS["krea-2"]))

    def test_flux2_beta_is_borrowed_not_matched(self):
        # Unlike krea-2/z-image (an exact VAE match), flux2's beta is only the
        # closest available analog -- FLUX.2's VAE has no measured fit here.
        # This just pins the documented relationship so a drift is caught.
        self.assertEqual(_speed_core._PRESETS["flux2"]["beta"],
                         _speed_core._PRESETS["flux"]["beta"])

    def test_delta_spans_a_usable_range(self):
        krea = _speed_core._PRESETS["krea-2"]
        self.assertAlmostEqual(self._sigma_star(krea, delta=0.005), 0.9273, places=4)
        self.assertAlmostEqual(self._sigma_star(krea, delta=0.05), 0.7989, places=4)


class TestKappaAndAlign(unittest.TestCase):
    def test_kappa_at_t0(self):
        self.assertAlmostEqual(kappa(0.0, 2.0), 2.0)

    def test_kappa_at_t1(self):
        self.assertAlmostEqual(kappa(1.0, 2.0), 1.0)

    def test_align_timestep(self):
        t = 0.5
        r = 2.0
        self.assertAlmostEqual(align_timestep(t, r), t * kappa(t, r))


class TestValidateScales(unittest.TestCase):
    def test_valid(self):
        validate_scales([0.25, 0.5, 1.0])

    def test_empty(self):
        with self.assertRaises(ValueError):
            validate_scales([])

    def test_not_ending_in_one(self):
        with self.assertRaises(ValueError):
            validate_scales([0.25, 0.75])

    def test_not_increasing(self):
        with self.assertRaises(ValueError):
            validate_scales([0.5, 0.25, 1.0])

    def test_out_of_range(self):
        with self.assertRaises(ValueError):
            validate_scales([0.0, 1.0])
        with self.assertRaises(ValueError):
            validate_scales([1.01, 1.0])


class TestDCTExpandNp(unittest.TestCase):
    def test_output_shape(self):
        x = np.random.randn(2, 4, 8, 8).astype(np.float32)
        out = _dct_expand_np(x, (16, 16), t=0.5, seed=42)
        self.assertEqual(out.shape, (2, 4, 16, 16))

    def test_rejects_smaller_target(self):
        x = np.random.randn(1, 4, 16, 16).astype(np.float32)
        with self.assertRaises(ValueError):
            _dct_expand_np(x, (8, 8), t=0.5, seed=42)

    def test_preserves_low_frequencies_approximately(self):
        x = np.random.randn(1, 1, 8, 8).astype(np.float32)
        out = _dct_expand_np(x, (16, 16), t=0.0, seed=42)
        # With t=0, the expanded high-freq coefficients are zero, so after
        # IDCT the energy should be concentrated in the low frequencies.
        coeffs_out = np.fft.fftshift(np.fft.fft2(out[0, 0], norm="ortho"))
        low_band_energy = np.sum(np.abs(coeffs_out[4:12, 4:12]) ** 2)
        total_energy = np.sum(np.abs(coeffs_out) ** 2)
        self.assertGreater(low_band_energy / total_energy, 0.8)


try:
    import pywt  # noqa: F401
    _HAS_PYWT = True
except ImportError:
    _HAS_PYWT = False


@unittest.skipUnless(_HAS_PYWT, "PyWavelets not installed (optional, dwt transform only)")
class TestDWTExpandNp(unittest.TestCase):
    def test_output_shape(self):
        x = np.random.randn(1, 4, 8, 8).astype(np.float32)
        out = _dwt_expand_np(x, t=0.5, seed=42)
        self.assertEqual(out.shape, (1, 4, 16, 16))

    def test_with_zero_noise_preserves_average(self):
        x = np.ones((1, 1, 4, 4), dtype=np.float32)
        out = _dwt_expand_np(x, t=0.0, seed=42)
        # With t=0, LH/HL/HH are all zero, so the result comes only from
        # LL band. The mean won't be exactly 2x due to wavelet
        # normalisation, but it should be proportional and non-zero.
        self.assertEqual(out.shape, (1, 1, 8, 8))
        self.assertGreater(float(out.mean()), 0.0)
        self.assertLess(abs(float(out.mean())), 10.0)


class TestFFTExpandNp(unittest.TestCase):
    def test_output_shape(self):
        x = np.random.randn(2, 3, 8, 8).astype(np.float32)
        out = _fft_expand_np(x, (16, 16), t=0.5, seed=42)
        self.assertEqual(out.shape, (2, 3, 16, 16))

    def test_rejects_smaller_target(self):
        x = np.random.randn(1, 4, 16, 16).astype(np.float32)
        with self.assertRaises(ValueError):
            _fft_expand_np(x, (8, 8), t=0.5, seed=42)


if __name__ == "__main__":
    unittest.main(verbosity=2)