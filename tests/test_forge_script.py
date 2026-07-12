"""Offline stub-harness tests for the Forge Neo script (scripts/speed_forge.py).

Fakes the webui ``modules`` package, ``gradio``, and ``k_diffusion`` so the
script's hook functions can be exercised directly with representative tensors,
without a running Forge instance.

Run from the repo root with:
  python -m unittest tests.test_forge_script -v
"""
import importlib.util as _iu
import pathlib
import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))


# =============================================================================
# Stub environment (must exist before scripts/speed_forge.py is imported)
# =============================================================================

_reported_errors = []


def _install_stubs():
    class _Script:
        pass

    scripts_stub = types.ModuleType("modules.scripts")
    scripts_stub.Script = _Script
    scripts_stub.AlwaysVisible = object()
    scripts_stub.basedir = lambda: str(_ROOT)

    errors_stub = types.ModuleType("modules.errors")
    errors_stub.report = lambda msg, **kw: _reported_errors.append(msg)

    ui_components_stub = types.ModuleType("modules.ui_components")
    ui_components_stub.InputAccordion = MagicMock()

    modules_stub = types.ModuleType("modules")
    modules_stub.scripts = scripts_stub
    modules_stub.errors = errors_stub
    modules_stub.ui_components = ui_components_stub

    kds_stub = types.ModuleType("k_diffusion.sampling")
    kds_stub.torch = torch  # replaced per-test with a fake hijack
    k_diffusion_stub = types.ModuleType("k_diffusion")
    k_diffusion_stub.sampling = kds_stub

    sys.modules.setdefault("gradio", MagicMock())
    sys.modules["modules"] = modules_stub
    sys.modules["modules.scripts"] = scripts_stub
    sys.modules["modules.errors"] = errors_stub
    sys.modules["modules.ui_components"] = ui_components_stub
    sys.modules["k_diffusion"] = k_diffusion_stub
    sys.modules["k_diffusion.sampling"] = kds_stub
    return kds_stub


_kds_stub = _install_stubs()

_spec = _iu.spec_from_file_location(
    "speed_forge_under_test", str(_ROOT / "scripts" / "speed_forge.py"))
speed_forge = _iu.module_from_spec(_spec)
sys.modules["speed_forge_under_test"] = speed_forge
_spec.loader.exec_module(speed_forge)

speed_core = speed_forge.speed_core


# =============================================================================
# Fakes
# =============================================================================

class RecordingSampler:
    """Fake k-diffusion solver: records calls, invokes the callback per step,
    optionally exercises randn_like, and returns x unchanged."""

    def __init__(self, use_randn=False):
        self.calls = []
        self.use_randn = use_randn
        self.__name__ = "sample_fake_euler"

    def __call__(self, model, x, sigmas, extra_args=None, callback=None,
                 disable=None, s_churn=0.0, noise_sampler=None):
        self.calls.append({
            "shape": tuple(x.shape),
            "sigmas": [float(s) for s in sigmas],
            "steps": len(sigmas) - 1,
            "noise_sampler": noise_sampler,
            "s_churn": s_churn,
        })
        for i in range(len(sigmas) - 1):
            if self.use_randn:
                kds = sys.modules["k_diffusion.sampling"]
                noise = kds.torch.randn_like(x)
                assert noise.shape == x.shape, (
                    f"randn_like shape {tuple(noise.shape)} != x {tuple(x.shape)}")
            if callback is not None:
                callback({"i": i, "x": x, "sigma": sigmas[i], "denoised": x})
        return x


class FakeFullResHijack:
    """Mimics Forge's TorchHijack: randn_like ignores the input shape and
    always returns a full-resolution tensor."""

    def __init__(self, full_shape):
        self.full_shape = full_shape

    def __getattr__(self, item):
        if item == "randn_like":
            return self.randn_like
        return getattr(torch, item)

    def randn_like(self, x):
        return torch.zeros(self.full_shape)


def make_p(sampler_func, is_hr_pass=False, mask=None, is_inpaint=False):
    return SimpleNamespace(
        sampler=SimpleNamespace(func=sampler_func),
        sd_model=SimpleNamespace(is_inpaint=is_inpaint),
        is_hr_pass=is_hr_pass,
        mask=mask,
        extra_generation_params={},
    )


UI_DEFAULTS = dict(transform="dct", mode="manual", model_preset="flux",
                   scales="0.5,1.0", delta=0.01, manual_sigmas="0.72",
                   spectrum_A=203.615097, spectrum_beta=1.915461,
                   seed=-1, apply_hr=False)


def run_hooks(p, enabled=True, seeds=(123,), **overrides):
    """Drive before_process_batch + process_before_every_sampling."""
    ui = {**UI_DEFAULTS, **overrides}
    script = speed_forge.SpeedForgeScript()
    script.before_process_batch(
        p, enabled, ui["transform"], ui["mode"], ui["model_preset"],
        ui["scales"], ui["delta"], ui["manual_sigmas"], ui["spectrum_A"],
        ui["spectrum_beta"], ui["seed"], ui["apply_hr"],
        seeds=list(seeds), prompts=["test"], subseeds=[0])
    script.process_before_every_sampling(p)
    return script


class ForgeScriptTestCase(unittest.TestCase):
    def setUp(self):
        _reported_errors.clear()
        cls = speed_forge.SpeedForgeScript
        cls.enabled = False
        cls.config = None
        cls.apply_hr = False
        cls.wrapped_passes = 0
        cls.completed_passes = 0
        _kds_stub.torch = torch


# =============================================================================
# Core segmentation tests (framework-agnostic path)
# =============================================================================

class TestCoreSegmentation(ForgeScriptTestCase):
    def test_two_segments_shapes_steps_and_callback(self):
        fake = RecordingSampler()
        sigmas = torch.linspace(1.0, 0.0, 21)
        x = torch.randn(1, 4, 32, 32)
        seen_steps = []

        out = speed_core.sample_speed_core(
            fake, None, x, sigmas, extra_args={},
            callback=lambda d: seen_steps.append(d["i"]), disable=True,
            transform="dct", mode="manual", scales=[0.5, 1.0],
            manual_sigmas=[0.72], seed=7,
        )

        self.assertEqual(len(fake.calls), 2)
        self.assertEqual(fake.calls[0]["shape"], (1, 4, 16, 16))
        self.assertEqual(fake.calls[1]["shape"], (1, 4, 32, 32))
        self.assertEqual(out.shape, (1, 4, 32, 32))
        # segments cover the full schedule
        self.assertEqual(fake.calls[0]["steps"] + fake.calls[1]["steps"], 20)
        # callback indices are re-based to the full schedule
        self.assertEqual(seen_steps, list(range(20)))
        # transition sigma was patched to the aligned timestep t*kappa(t, 2)
        t = fake.calls[0]["sigmas"][-1]
        expected = t * 2.0 / (1.0 + t)
        self.assertAlmostEqual(fake.calls[1]["sigmas"][0], expected, places=5)

    def test_5d_video_latent(self):
        fake = RecordingSampler()
        sigmas = torch.linspace(1.0, 0.0, 11)
        x = torch.randn(1, 4, 3, 16, 16)

        out = speed_core.sample_speed_core(
            fake, None, x, sigmas, extra_args={}, disable=True,
            transform="dct", mode="manual", scales=[0.5, 1.0],
            manual_sigmas=[0.55], seed=1,
        )

        self.assertEqual(fake.calls[0]["shape"], (1, 4, 3, 8, 8))
        self.assertEqual(out.shape, (1, 4, 3, 16, 16))

    def test_full_res_kwargs_only_on_final_segment(self):
        fake = RecordingSampler()
        sigmas = torch.linspace(1.0, 0.0, 21)
        sentinel = object()

        speed_core.sample_speed_core(
            fake, None, torch.randn(1, 4, 32, 32), sigmas, extra_args={},
            disable=True, transform="dct", mode="manual", scales=[0.5, 1.0],
            manual_sigmas=[0.72], seed=0,
            full_res_sampler_kwargs={"noise_sampler": sentinel},
        )

        self.assertIsNone(fake.calls[0]["noise_sampler"])
        self.assertIs(fake.calls[1]["noise_sampler"], sentinel)

    def test_delta_optimal_flux_preset_transitions(self):
        fake = RecordingSampler()
        sigmas = torch.linspace(1.0, 0.0, 21)

        out = speed_core.sample_speed_core(
            fake, None, torch.randn(1, 4, 64, 64), sigmas, extra_args={},
            disable=True, transform="dct", mode="delta_optimal",
            scales=[0.5, 1.0], delta=0.01,
            spectrum_A=203.615097, spectrum_beta=1.915461, seed=0,
        )

        self.assertEqual(len(fake.calls), 2)
        self.assertEqual(fake.calls[0]["shape"], (1, 4, 32, 32))
        self.assertEqual(out.shape, (1, 4, 64, 64))


# =============================================================================
# Script hook / wrapper tests
# =============================================================================

class TestScriptHooks(ForgeScriptTestCase):
    def test_wraps_sampler_and_runs(self):
        fake = RecordingSampler(use_randn=True)
        p = make_p(fake)
        full_shape = (1, 4, 32, 32)
        hijack = FakeFullResHijack(full_shape)
        _kds_stub.torch = hijack

        script = run_hooks(p, seeds=(42,))
        self.assertTrue(getattr(p.sampler.func, "_speed_wrapped", False))
        self.assertEqual(speed_forge.SpeedForgeScript.wrapped_passes, 1)
        self.assertEqual(speed_forge.SpeedForgeScript.config["seed"], 42)
        self.assertIn("SPEED", p.extra_generation_params)

        # signature is inherited from the base sampler (functools.wraps)
        import inspect
        self.assertIn("s_churn", inspect.signature(p.sampler.func).parameters)

        sigmas = torch.linspace(1.0, 0.0, 21)
        sentinel = object()
        out = p.sampler.func(None, torch.randn(*full_shape), sigmas=sigmas,
                             extra_args={}, callback=None, disable=True,
                             s_churn=0.25, noise_sampler=sentinel)

        # two segments ran; randn_like shape assertions inside the fake passed,
        # including on the coarse segment where the hijack returns full-res noise
        self.assertEqual(len(fake.calls), 2)
        self.assertEqual(fake.calls[0]["shape"], (1, 4, 16, 16))
        self.assertEqual(out.shape, full_shape)
        # extra sampler kwargs reach every segment; noise_sampler only full-res
        self.assertEqual(fake.calls[0]["s_churn"], 0.25)
        self.assertIsNone(fake.calls[0]["noise_sampler"])
        self.assertIs(fake.calls[1]["noise_sampler"], sentinel)
        # the hijack is restored after the pass
        self.assertIs(_kds_stub.torch, hijack)
        self.assertEqual(speed_forge.SpeedForgeScript.completed_passes, 1)

        script.postprocess(p, None)
        self.assertFalse(speed_forge.SpeedForgeScript.enabled)
        self.assertIsNone(speed_forge.SpeedForgeScript.config)

    def test_disabled_leaves_sampler_untouched(self):
        fake = RecordingSampler()
        p = make_p(fake)
        run_hooks(p, enabled=False)
        self.assertIs(p.sampler.func, fake)

    def test_inpaint_skipped(self):
        fake = RecordingSampler()
        p = make_p(fake, mask=torch.ones(1))
        run_hooks(p)
        self.assertIs(p.sampler.func, fake)

    def test_hr_pass_gated_by_checkbox(self):
        fake = RecordingSampler()
        p = make_p(fake, is_hr_pass=True)
        run_hooks(p, apply_hr=False)
        self.assertIs(p.sampler.func, fake)

        p2 = make_p(RecordingSampler(), is_hr_pass=True)
        run_hooks(p2, apply_hr=True)
        self.assertTrue(getattr(p2.sampler.func, "_speed_wrapped", False))

    def test_no_double_wrap(self):
        fake = RecordingSampler()
        p = make_p(fake)
        script = run_hooks(p)
        first = p.sampler.func
        script.process_before_every_sampling(p)
        self.assertIs(p.sampler.func, first)

    def test_invalid_scales_disable_run(self):
        fake = RecordingSampler()
        p = make_p(fake)
        run_hooks(p, scales="0.5,0.9")  # does not end at 1.0
        self.assertIs(p.sampler.func, fake)
        self.assertFalse(speed_forge.SpeedForgeScript.enabled)
        self.assertEqual(len(_reported_errors), 1)

    def test_manual_sigmas_above_one_allowed(self):
        fake = RecordingSampler()
        p = make_p(fake)
        run_hooks(p, scales="0.25,0.5,1.0", manual_sigmas="3.5,0.72")
        self.assertTrue(getattr(p.sampler.func, "_speed_wrapped", False))

    def test_explicit_seed_used(self):
        p = make_p(RecordingSampler())
        run_hooks(p, seed=999, seeds=(42,))
        self.assertEqual(speed_forge.SpeedForgeScript.config["seed"], 999)


if __name__ == "__main__":
    unittest.main(verbosity=2)
