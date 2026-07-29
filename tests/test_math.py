"""Math sanity tests for SPEED spectral utilities.

Based on https://github.com/howardhx/speed
Run from the repo root with:
  python -m unittest tests.test_math -v
"""
import importlib.util as _iu
import math
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
        "flux": 0.9680,        # measured
        "wan21": 0.6941,
        "krea-2": 0.9000,
        "krea-2-raw": 0.8300,
        "z-image": 0.9400,
        "flux2": 0.9680,       # measured
    }

    # The sigma anchor both measured presets share. Everything at or above this
    # passed A/B on two models, two resolutions and two schedules; 0.9389 is the
    # nearest measured failure (Flux.1 license-plate text at 1024x1024).
    MEASURED_ANCHOR = 0.9680

    # Share of the schedule spent coarse at the reference. Only meaningful for
    # presets still on the spectrum-derived threshold, which are the ones that
    # keep the step-fraction cap.
    REFERENCE_COARSE_SHARE = {
        "wan21": 0.7790,
        "krea-2": 0.2598,
        "krea-2-raw": 0.3928,
        "z-image": 0.1607,
    }

    def _sigma_star(self, preset, scale=0.5, delta=0.01, latent=None):
        # Default to the preset's own reference geometry -- flux2's differs.
        latent = preset["ref_latent"] if latent is None else latent
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

    def test_every_preset_declares_its_reference_latent(self):
        for name, preset in _speed_core._PRESETS.items():
            if preset is None:
                continue
            with self.subTest(preset=name):
                self.assertIn("ref_latent", preset)

    def test_flux2_reference_latent_is_half(self):
        # FLUX.2 encodes at 8x into 32 channels then packs 2x2 spatially into
        # the channel axis, so its sampler-visible latent is 1/16 scale: a
        # 1024px image is 64x64, not 128x128. Its sigma* is therefore
        # calibrated at omega_max 32, not 64.
        self.assertEqual(_speed_core._PRESETS["flux2"]["ref_latent"], 64)
        for name in ("flux", "wan21", "krea-2", "krea-2-raw", "z-image"):
            with self.subTest(preset=name):
                self.assertEqual(_speed_core._PRESETS[name]["ref_latent"], 128)

    def test_flux2_sigma_star_holds_at_its_own_geometry(self):
        # Guards the recalibration: assuming the 8x geometry (a 128 latent at
        # 1024px) puts the same A at a much higher Nyquist frequency, where the
        # spectrum is weaker and the transition lands far later than the target
        # -- which is exactly the bug this preset shipped with first.
        preset = _speed_core._PRESETS["flux2"]
        self.assertAlmostEqual(self._sigma_star(preset, latent=64),
                               self.MEASURED_ANCHOR, places=4)
        self.assertLess(self._sigma_star(preset, latent=128), self.MEASURED_ANCHOR)

    def test_documented_reference_coarse_shares(self):
        # Only the spectrum-derived presets carry a share; the measured ones
        # deliberately do not (their sigma is the answer).
        for name, expected in self.REFERENCE_COARSE_SHARE.items():
            with self.subTest(preset=name):
                preset = _speed_core._PRESETS[name]
                share = _spectral.reference_coarse_fraction(
                    self._sigma_star(preset), preset["ref_shift"])
                self.assertAlmostEqual(share, expected, places=4)

    def test_measured_presets_carry_no_ref_shift(self):
        # ref_shift enables the step-fraction cap, which exists to compensate
        # for an uncalibrated threshold. A measured threshold must not be
        # overridden by it -- for flux the cap would demand 5 of 40 steps where
        # the measured-correct answer is 11.
        for name in ("flux", "flux2"):
            with self.subTest(preset=name):
                self.assertNotIn("ref_shift", _speed_core._PRESETS[name])
        for name in ("wan21", "krea-2", "krea-2-raw", "z-image"):
            with self.subTest(preset=name):
                self.assertIn("ref_shift", _speed_core._PRESETS[name])

    def test_measured_presets_share_the_sigma_anchor(self):
        # The quality boundary is a sigma and it transferred across models, so
        # both calibrated presets pin the same one despite different geometry.
        for name in ("flux", "flux2"):
            with self.subTest(preset=name):
                self.assertAlmostEqual(self._sigma_star(_speed_core._PRESETS[name]),
                                       self.MEASURED_ANCHOR, places=4)
        self.assertNotEqual(_speed_core._PRESETS["flux"]["ref_latent"],
                            _speed_core._PRESETS["flux2"]["ref_latent"])

    def test_anchor_sits_above_the_measured_failure(self):
        # Flux.1 at 1024x1024 rendered text correctly transitioning at 0.9611
        # and incorrectly at 0.9389. The anchor must clear the failure.
        self.assertGreater(self.MEASURED_ANCHOR, 0.9611)

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


class TestAdaptiveDelta(unittest.TestCase):
    """``adaptive_delta`` holds the coarse/detail step split across resolutions.

    Schedules here are the flow-matching ones these presets target: sigma =
    exp(mu)*t / (exp(mu)*t + 1 - t) over a uniform t, which is what
    ``PredictionFlux`` / ComfyUI's ``ModelSamplingFlux`` produce.
    """

    A, BETA = 203.615097, 1.915461   # flux preset

    def _schedule(self, steps, mu=1.15):
        shift = math.exp(mu)
        ts = [1.0 - i / steps for i in range(steps)]
        return torch.tensor([shift * t / (shift * t + 1.0 - t) for t in ts] + [0.0])

    REF_SHIFT = 3.158193                  # flux: exp(mu=1.15)

    def _coarse_steps(self, latent, steps=40, mu=1.15, adaptive=False, scales=None,
                      ref_shift=None):
        sigmas = self._schedule(steps, mu)
        transitions = _speed_core._resolve_transitions(
            sigmas, scales or [0.5, 1.0], 0.01, self.A, self.BETA, latent, latent,
            adaptive=adaptive, ref_latent=128, ref_shift=ref_shift)
        return transitions[-1][0] if transitions else steps

    def test_without_adaptive_the_coarse_share_grows_with_resolution(self):
        # The reported problem: same preset and delta, bigger canvas, more of
        # the schedule spent painting the coarse pass.
        at_1024 = self._coarse_steps(128)
        at_2048 = self._coarse_steps(256)
        self.assertGreater(at_2048, at_1024)

    def test_adaptive_pins_the_split_across_resolutions(self):
        reference = self._coarse_steps(128, adaptive=True)
        for latent in (192, 256, 384):    # 1536, 2048, 3072 px at 8x
            with self.subTest(latent=latent):
                self.assertEqual(self._coarse_steps(latent, adaptive=True), reference)

    def test_adaptive_is_a_noop_at_and_below_the_reference(self):
        # Never more aggressive than the unadapted path: below the reference the
        # actual threshold is already the conservative one and is kept.
        for latent in (64, 96, 128):      # 512, 768, 1024 px
            with self.subTest(latent=latent):
                self.assertEqual(self._coarse_steps(latent, adaptive=True),
                                 self._coarse_steps(latent, adaptive=False))

    def test_adaptive_never_transitions_later_than_unadapted(self):
        for latent in (64, 128, 192, 256, 384):
            with self.subTest(latent=latent):
                self.assertLessEqual(self._coarse_steps(latent, adaptive=True),
                                     self._coarse_steps(latent, adaptive=False))

    def test_cap_rescues_a_resolution_dependent_schedule_shift(self):
        # ComfyUI ModelSamplingFlux / SwarmUI Sigma Shift derive mu from
        # width*height, which compresses sigma toward 1.0 and pushes the
        # crossing later no matter what the threshold is. Unguarded at 3072px
        # the transition never fires at all and the output stays half-size.
        steps, mu_3072 = 40, 6.70
        self.assertEqual(self._coarse_steps(384, steps, mu_3072, adaptive=False), steps)
        capped = self._coarse_steps(384, steps, mu_3072, adaptive=True)
        self.assertLessEqual(capped, steps * _speed_core.MAX_COARSE_FRACTION)
        self.assertGreater(steps - capped, 0)

    def test_shifted_schedule_reproduces_the_reference_split_not_a_flat_cap(self):
        # For a preset still on the spectrum-derived threshold, a resolution
        # dependent schedule should land on the *reference* split rather than
        # wherever the sigma alone crosses.
        steps = 40
        reference = self._coarse_steps(128, steps, adaptive=True, ref_shift=self.REF_SHIFT)
        for mu in (2.02, 3.23, 6.70):          # 1536, 2048, 3072 px with Sigma Shift
            with self.subTest(mu=mu):
                derived = self._coarse_steps(192, steps, mu, adaptive=True,
                                             ref_shift=self.REF_SHIFT)
                self.assertEqual(derived, reference)
                # Without ref_shift the sigma governs and lands much later.
                unheld = self._coarse_steps(192, steps, mu, adaptive=True)
                self.assertGreater(unheld, derived)

    def test_derived_share_wins_over_the_flat_backstop(self):
        # The flat fraction is arbitrary; the derived share is not. When a
        # preset declares ref_shift the derived value must be used even where
        # it is the more permissive of the two, otherwise a large delta silently
        # stops meaning what it says. delta 0.5 is chosen because its derived
        # share (34/40) sits above the backstop (32/40).
        steps, delta = 40, 0.5
        sigmas = self._schedule(steps, mu=6.70)
        transitions = _speed_core._resolve_transitions(
            sigmas, [0.5, 1.0], delta, self.A, self.BETA, 192, 192,
            adaptive=True, ref_latent=128, ref_shift=self.REF_SHIFT)
        t_ref = delta_optimal_transitions([0.5, 1.0], delta, self.A, self.BETA, 128, 128)[0]
        expected = math.ceil(steps * _spectral.reference_coarse_fraction(t_ref, self.REF_SHIFT))
        self.assertGreater(expected, steps * _speed_core.MAX_COARSE_FRACTION)
        self.assertEqual(transitions[-1][0], expected)

    def test_flat_backstop_still_applies_without_a_ref_shift(self):
        # `custom` has no model to derive a share from, so the arbitrary bound
        # is all that is left and must stay active.
        steps = 40
        coarse = self._coarse_steps(384, steps, mu=6.70, adaptive=True, ref_shift=None)
        self.assertEqual(coarse, int(steps * _speed_core.MAX_COARSE_FRACTION))

    def test_reference_cap_does_not_bind_at_the_reference(self):
        # Deriving the cap must not disturb the case it is derived from.
        for latent in (64, 96, 128):
            with self.subTest(latent=latent):
                self.assertEqual(
                    self._coarse_steps(latent, adaptive=True, ref_shift=self.REF_SHIFT),
                    self._coarse_steps(latent, adaptive=False))

    def test_cap_does_not_bind_on_a_designed_configuration(self):
        # The safety net must be invisible in normal use.
        for latent in (128, 192, 256):
            with self.subTest(latent=latent):
                coarse = self._coarse_steps(latent, adaptive=True)
                self.assertLess(coarse, 40 * _speed_core.MAX_COARSE_FRACTION)

    def test_multi_scale_ladder_stays_ordered_under_the_cap(self):
        transitions = _speed_core._resolve_transitions(
            self._schedule(40, mu=6.70), [0.25, 0.5, 1.0], 0.01, self.A, self.BETA,
            384, 384, adaptive=True, ref_latent=128)
        indices = [idx for idx, _, _ in transitions]
        self.assertEqual(indices, sorted(indices))
        self.assertLessEqual(indices[-1], 40 * _speed_core.MAX_COARSE_FRACTION)

    def test_sigma_divisor_of_one_is_an_exact_noop(self):
        # neo_shift's whole selling point: divisor=1.0 must reproduce plain
        # delta_optimal exactly, not just approximately.
        sigmas = self._schedule(40)
        with_default = _speed_core._resolve_transitions(
            sigmas, [0.5, 1.0], 0.01, self.A, self.BETA, 128, 128,
            adaptive=True, ref_latent=128, sigma_divisor=1.0)
        without = _speed_core._resolve_transitions(
            sigmas, [0.5, 1.0], 0.01, self.A, self.BETA, 128, 128,
            adaptive=True, ref_latent=128)
        self.assertEqual(with_default, without)

    def test_higher_sigma_divisor_gives_more_coarse_steps(self):
        # Divisor > 1 lowers the threshold -> transitions later -> more speed.
        baseline = self._coarse_steps(128, adaptive=True)
        sigmas = self._schedule(40)
        divided = _speed_core._resolve_transitions(
            sigmas, [0.5, 1.0], 0.01, self.A, self.BETA, 128, 128,
            adaptive=True, ref_latent=128, sigma_divisor=1.03)
        self.assertGreater(divided[-1][0], baseline)

    def test_lower_sigma_divisor_gives_fewer_coarse_steps(self):
        # Divisor < 1 raises the threshold -> transitions earlier -> more conservative.
        baseline = self._coarse_steps(128, adaptive=True)
        sigmas = self._schedule(40)
        divided = _speed_core._resolve_transitions(
            sigmas, [0.5, 1.0], 0.01, self.A, self.BETA, 128, 128,
            adaptive=True, ref_latent=128, sigma_divisor=0.9)
        self.assertLess(divided[-1][0], baseline)

    def test_sigma_divisor_applies_regardless_of_adaptive(self):
        # neo_shift is "delta_optimal plus a divisor" either way adaptive_delta
        # is set -- the divide must not be gated on the adaptive branch.
        sigmas = self._schedule(40)
        without_adaptive = _speed_core._resolve_transitions(
            sigmas, [0.5, 1.0], 0.01, self.A, self.BETA, 128, 128,
            adaptive=False, sigma_divisor=1.0)
        divided = _speed_core._resolve_transitions(
            sigmas, [0.5, 1.0], 0.01, self.A, self.BETA, 128, 128,
            adaptive=False, sigma_divisor=1.03)
        self.assertGreater(divided[-1][0], without_adaptive[-1][0])

    def test_sigma_divisor_applies_to_every_transition(self):
        sigmas = self._schedule(40)
        baseline = _speed_core._resolve_transitions(
            sigmas, [0.25, 0.5, 1.0], 0.01, self.A, self.BETA, 128, 128,
            adaptive=True, ref_latent=128)
        divided = _speed_core._resolve_transitions(
            sigmas, [0.25, 0.5, 1.0], 0.01, self.A, self.BETA, 128, 128,
            adaptive=True, ref_latent=128, sigma_divisor=1.03)
        self.assertEqual(len(divided), len(baseline))
        for i in range(len(baseline)):
            with self.subTest(transition=i):
                self.assertGreaterEqual(divided[i][0], baseline[i][0])

    def test_sigma_divisor_logs_only_when_not_a_noop(self):
        sigmas = self._schedule(40)
        logged = []
        _speed_core._resolve_transitions(
            sigmas, [0.5, 1.0], 0.01, self.A, self.BETA, 128, 128,
            adaptive=True, ref_latent=128, sigma_divisor=1.0, log_fn=logged.append)
        self.assertFalse(any("neo_shift" in m for m in logged))

        logged.clear()
        _speed_core._resolve_transitions(
            sigmas, [0.5, 1.0], 0.01, self.A, self.BETA, 128, 128,
            adaptive=True, ref_latent=128, sigma_divisor=1.03, log_fn=logged.append)
        self.assertTrue(any("neo_shift" in m for m in logged))

    def test_equivalent_delta_round_trips(self):
        # The number reported in the adaptive log line must actually reproduce
        # the pinned threshold at the frequency it is quoted for.
        for omega in (16.0, 32.0, 48.0, 96.0):
            with self.subTest(omega=omega):
                target = 0.8527
                d = _spectral.equivalent_delta(target, self.A, self.BETA, omega)
                back = activation_time(power_spectrum(omega, self.A, self.BETA), d)
                self.assertAlmostEqual(back, target, places=6)


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
        # Seeded deliberately: on 8x8 white noise this ratio ranges roughly
        # 0.67-0.92 (mean 0.83) across inputs, so an unseeded draw fails the
        # 0.8 threshold about a quarter of the time. The property is real, the
        # sample is just small -- pin the input rather than loosen the bound.
        x = np.random.default_rng(0).standard_normal((1, 1, 8, 8)).astype(np.float32)
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