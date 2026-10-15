"""Offline stub-harness tests for the Forge Neo script (scripts/speed_forge.py).

Fakes the webui ``modules`` package, ``gradio``, and ``k_diffusion`` so the
script's hook functions can be exercised directly with representative tensors,
without a running Forge instance.

Run from the repo root with:
  python -m unittest tests.test_forge_script -v
"""
import contextlib
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

    prompt_parser_stub = types.ModuleType("modules.prompt_parser")

    class SdConditioning(list):
        def __init__(self, prompts, **kw):
            super().__init__(prompts)
            self.kw = kw

    prompt_parser_stub.SdConditioning = SdConditioning
    prompt_parser_stub.get_learned_conditioning = lambda *a, **kw: "encoded"

    shared_stub = types.ModuleType("modules.shared")
    shared_stub.opts = SimpleNamespace(skip_early_cond=0.0)

    modules_stub = types.ModuleType("modules")
    modules_stub.scripts = scripts_stub
    modules_stub.errors = errors_stub
    modules_stub.ui_components = ui_components_stub
    modules_stub.prompt_parser = prompt_parser_stub
    modules_stub.shared = shared_stub

    kds_stub = types.ModuleType("k_diffusion.sampling")
    kds_stub.torch = torch  # replaced per-test with a fake hijack
    k_diffusion_stub = types.ModuleType("k_diffusion")
    k_diffusion_stub.sampling = kds_stub

    sys.modules.setdefault("gradio", MagicMock())
    sys.modules["modules"] = modules_stub
    sys.modules["modules.scripts"] = scripts_stub
    sys.modules["modules.errors"] = errors_stub
    sys.modules["modules.ui_components"] = ui_components_stub
    sys.modules["modules.prompt_parser"] = prompt_parser_stub
    sys.modules["modules.shared"] = shared_stub
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

    def __init__(self, use_randn=False, raise_at_call=None):
        self.calls = []
        self.use_randn = use_randn
        self.raise_at_call = raise_at_call
        self.__name__ = "sample_fake_euler"

    def __call__(self, model, x, sigmas, extra_args=None, callback=None,
                 disable=None, s_churn=0.0, noise_sampler=None):
        self.calls.append({
            "shape": tuple(x.shape),
            "sigmas": [float(s) for s in sigmas],
            "steps": len(sigmas) - 1,
            "noise_sampler": noise_sampler,
            "s_churn": s_churn,
            "extra_args": extra_args,
            "cond_scale": (extra_args or {}).get("cond_scale"),
            "uncond": (extra_args or {}).get("uncond"),
            "total_steps": getattr(model, "total_steps", None),
            "turbo_strengths": [float(o.patch[0][0]) for o in _TURBO_OBJECTS],
        })
        if self.raise_at_call is not None and len(self.calls) == self.raise_at_call:
            raise RuntimeError("interrupted")
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


class _FakeAnimaModel:
    """Stands in for Forge's Anima diffusion engine.

    The guard identifies the model by class name, which is what the real engine
    (``backend/diffusion_engine/anima.py``) exposes without importing it.
    """
    is_inpaint = False


_FakeAnimaModel.__name__ = "Anima"


@contextlib.contextmanager
def _fake_backend_args(ref_latents):
    """Install a stub ``backend.args`` exposing ``dynamic_args.ref_latents``.

    Installed per-test rather than globally so the default path still exercises
    the ImportError branch (no Forge backend present).
    """
    args_stub = types.ModuleType("backend.args")
    args_stub.dynamic_args = SimpleNamespace(ref_latents=list(ref_latents))
    backend_stub = types.ModuleType("backend")
    backend_stub.args = args_stub
    saved = {k: sys.modules.get(k) for k in ("backend", "backend.args")}
    sys.modules["backend"] = backend_stub
    sys.modules["backend.args"] = args_stub
    try:
        yield args_stub
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


def make_p(sampler_func, is_hr_pass=False, mask=None, is_inpaint=False):
    return SimpleNamespace(
        sampler=SimpleNamespace(func=sampler_func),
        sd_model=SimpleNamespace(is_inpaint=is_inpaint),
        is_hr_pass=is_hr_pass,
        mask=mask,
        extra_generation_params={},
    )


UI_DEFAULTS = dict(transform="dct", mode="manual", model_preset="flux",
                   scales="0.5,1.0", adaptive_delta=True, delta=0.01,
                   sigma_divisor=1.03, manual_sigmas="0.72",
                   spectrum_A=203.615097, spectrum_beta=1.915461, seed=-1,
                   apply_hr=False, diversity_enabled=False, diversity_ref_steps=28,
                   diversity_cfg=4.5, diversity_pattern="turbo")


def run_hooks(p, enabled=True, seeds=(123,), uc=None, **overrides):
    """Drive before_process_batch + process_before_every_sampling."""
    ui = {**UI_DEFAULTS, **overrides}
    script = speed_forge.SpeedForgeScript()
    script.before_process_batch(
        p, enabled, ui["transform"], ui["mode"], ui["model_preset"],
        ui["scales"], ui["adaptive_delta"], ui["delta"], ui["sigma_divisor"],
        ui["manual_sigmas"], ui["spectrum_A"], ui["spectrum_beta"], ui["seed"],
        ui["apply_hr"], ui["diversity_enabled"], ui["diversity_ref_steps"],
        ui["diversity_cfg"], ui["diversity_pattern"],
        seeds=list(seeds), prompts=["test"], subseeds=[0])
    script.process_before_every_sampling(p, x=None, noise=None, c="cond", uc=uc)
    return script


# =============================================================================
# RAW composition stage fakes
# =============================================================================

KREA_SHIFT = 3.158193


def flow_schedule(n, shift=KREA_SHIFT):
    """Uniform-in-t flow schedule with the exponential shift, ending at 0."""
    t = torch.linspace(1.0, 0.0, n + 1)
    return shift * t / (1.0 + (shift - 1.0) * t)


class _FakeKrea2:
    is_inpaint = False


_FakeKrea2.__name__ = "Krea2"


class _FakeZImage:
    is_inpaint = False


_FakeZImage.__name__ = "ZImage"

# Turbo LoRA patch objects whose strength RecordingSampler records per call.
_TURBO_OBJECTS = []


def make_div_p(sampler_func, model_cls=_FakeKrea2, is_hr_pass=False):
    """A fake ``p`` carrying what the RAW stage reads."""
    encodes = []

    def get_conds_with_caching(function, required_prompts, steps, caches,
                               extra_network_data, hires_steps=None):
        encodes.append(dict(function=function, prompts=required_prompts, steps=steps,
                            caches=caches, extra_network_data=extra_network_data))
        return "UC"

    reference_calls = []

    def get_sigmas(processing, steps):
        reference_calls.append(steps)
        return flow_schedule(steps)

    p = SimpleNamespace(
        sampler=SimpleNamespace(func=sampler_func, get_sigmas=get_sigmas),
        sd_model=model_cls(), is_hr_pass=is_hr_pass, mask=None,
        extra_generation_params={}, negative_prompts=["blurry"], width=1024,
        height=1024, distilled_cfg_scale=3.5, firstpass_steps=12,
        cached_uc=[None, None], extra_network_data={"lora": []},
        get_conds_with_caching=get_conds_with_caching, s_min_uncond=0.0,
    )
    p.encodes = encodes
    p.reference_calls = reference_calls
    return p


DIV_UI = dict(diversity_enabled=True, scales="0.5,1.0", manual_sigmas="0.9")

ORIGINAL_ARGS = {"cond": "cond", "uncond": None, "cond_scale": 1.0,
                 "image_cond": None, "s_min_uncond": 0.0}


def run_div_pass(p, sigmas, model=None, seen=None, **ui):
    """Hooks plus one wrapped sampling pass; returns the sampler output."""
    run_hooks(p, **{**DIV_UI, **ui})
    return p.sampler.func(model, torch.randn(1, 4, 32, 32), sigmas=sigmas,
                          extra_args=ORIGINAL_ARGS, disable=True,
                          callback=None if seen is None else (lambda d: seen.append(d["i"])))


class ForgeScriptTestCase(unittest.TestCase):
    def setUp(self):
        _reported_errors.clear()
        cls = speed_forge.SpeedForgeScript
        cls.enabled = False
        cls.config = None
        cls.apply_hr = False
        cls.wrapped_passes = 0
        cls.completed_passes = 0
        cls.diversity = None
        _kds_stub.torch = torch
        sys.modules["modules.shared"].opts.skip_early_cond = 0.0


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

    def test_anima_reference_latents_skipped(self):
        # Anima concatenates its reference latents onto x along the temporal
        # axis, which requires them to match x spatially -- a coarse segment
        # breaks that. The guard keys off the runtime list, not the setting.
        fake = RecordingSampler()
        p = make_p(fake)
        p.sd_model = _FakeAnimaModel()
        with _fake_backend_args(ref_latents=[torch.zeros(1, 16, 1, 128, 128)]):
            run_hooks(p)
        self.assertIs(p.sampler.func, fake)

    def test_anima_without_references_still_wrapped(self):
        # Anima Edit turned off (or simply no references on this run) is the
        # ordinary case and must not lose SPEED.
        p = make_p(RecordingSampler())
        p.sd_model = _FakeAnimaModel()
        with _fake_backend_args(ref_latents=[]):
            run_hooks(p)
        self.assertTrue(getattr(p.sampler.func, "_speed_wrapped", False))

    def test_reference_latents_on_other_models_not_skipped(self):
        # Flux/Krea/Qwen populate the same list but consume it differently;
        # the guard is deliberately scoped to Anima.
        p = make_p(RecordingSampler())
        with _fake_backend_args(ref_latents=[torch.zeros(1, 16, 1, 128, 128)]):
            run_hooks(p)
        self.assertTrue(getattr(p.sampler.func, "_speed_wrapped", False))

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


# =============================================================================
# RAW composition stage (diversity subsection)
# =============================================================================

class TestDiversityStage(ForgeScriptTestCase):
    def setUp(self):
        super().setUp()
        self.sigmas = flow_schedule(12)
        self.ref = [float(v) for v in flow_schedule(28)]
        # first user step at or below 0.9 is the transition
        self.t_idx = next(i for i, v in enumerate(self.sigmas) if float(v) <= 0.9)
        self.t_sigma = float(self.sigmas[self.t_idx])

    def test_coarse_redensified_with_cfg_fine_untouched(self):
        fake = RecordingSampler()
        p = make_div_p(fake)
        run_div_pass(p, self.sigmas)

        self.assertEqual(len(fake.calls), 2)
        coarse, fine = fake.calls
        expected = ([1.0] + [r for r in self.ref if self.t_sigma + 1e-4 < r < 1.0 - 1e-4]
                    + [self.t_sigma])
        self.assertEqual(len(coarse["sigmas"]), len(expected))
        for a, b in zip(coarse["sigmas"], expected):
            self.assertAlmostEqual(a, b, places=5)
        self.assertGreater(coarse["steps"], self.t_idx)
        self.assertEqual(coarse["cond_scale"], 4.5)
        self.assertEqual(coarse["uncond"], "UC")
        self.assertIsNot(coarse["extra_args"], ORIGINAL_ARGS)
        self.assertEqual(coarse["extra_args"]["cond"], "cond")
        # the fine segment gets Forge's own dict, unmodified
        self.assertIs(fine["extra_args"], ORIGINAL_ARGS)
        self.assertEqual(ORIGINAL_ARGS["cond_scale"], 1.0)
        self.assertIsNone(ORIGINAL_ARGS["uncond"])
        self.assertEqual(fine["steps"], 12 - self.t_idx)
        self.assertEqual(p.reference_calls, [28])

    def test_negative_encoded_exactly_like_setup_conds(self):
        fake = RecordingSampler()
        p = make_div_p(fake)
        run_div_pass(p, self.sigmas)
        self.assertEqual(len(p.encodes), 1)
        enc = p.encodes[0]
        pp = sys.modules["modules.prompt_parser"]
        self.assertIs(enc["function"], pp.get_learned_conditioning)
        self.assertEqual(list(enc["prompts"]), ["blurry"])
        self.assertEqual(enc["prompts"].kw, dict(width=1024, height=1024,
                                                 is_negative_prompt=True,
                                                 distilled_cfg_scale=3.5))
        self.assertEqual(enc["steps"], 12)
        self.assertIs(enc["caches"][0], p.cached_uc)
        self.assertIs(enc["extra_network_data"], p.extra_network_data)

    def test_existing_uc_is_reused(self):
        fake = RecordingSampler()
        p = make_div_p(fake)
        run_hooks(p, uc="FORGE_UC", **DIV_UI)
        p.sampler.func(None, torch.randn(1, 4, 32, 32), sigmas=self.sigmas,
                       extra_args=ORIGINAL_ARGS, disable=True)
        self.assertEqual(p.encodes, [])
        self.assertEqual(fake.calls[0]["uncond"], "FORGE_UC")

    def test_callback_indices_stay_on_the_user_schedule(self):
        seen = []
        fake = RecordingSampler()
        run_div_pass(make_div_p(fake), self.sigmas, seen=seen)
        self.assertGreater(fake.calls[0]["steps"], self.t_idx)
        self.assertEqual(seen, list(range(12)))

    def test_total_steps_bumped_for_the_pass_and_restored(self):
        fake = RecordingSampler()
        model = SimpleNamespace(steps=12, total_steps=12)
        run_div_pass(make_div_p(fake), self.sigmas, model=model)
        extra = fake.calls[0]["steps"] - self.t_idx
        self.assertEqual(fake.calls[0]["total_steps"], 12 + extra)
        self.assertEqual(fake.calls[1]["total_steps"], 12 + extra)
        self.assertEqual(model.total_steps, 12)

    def test_total_steps_scaled_for_second_order_samplers(self):
        fake = RecordingSampler()
        model = SimpleNamespace(steps=12, total_steps=24)
        run_div_pass(make_div_p(fake), self.sigmas, model=model)
        extra = fake.calls[0]["steps"] - self.t_idx
        self.assertEqual(fake.calls[0]["total_steps"], 24 + 2 * extra)
        self.assertEqual(model.total_steps, 24)

    def test_total_steps_restored_after_interruption(self):
        fake = RecordingSampler(raise_at_call=1)
        model = SimpleNamespace(steps=12, total_steps=12)
        with self.assertRaises(RuntimeError):
            run_div_pass(make_div_p(fake), self.sigmas, model=model)
        self.assertEqual(model.total_steps, 12)

    def test_infotext_extends_the_speed_string(self):
        fake = RecordingSampler()
        p = make_div_p(fake)
        run_div_pass(p, self.sigmas)
        info = p.extra_generation_params["SPEED"]
        self.assertTrue(info.startswith("dct/manual/flux scales=0.5,1.0"), info)
        self.assertTrue(info.endswith(
            f" diversity(ref=28, cfg=4.5, coarse={fake.calls[0]['steps']})"), info)

    def test_experimental_models_run_and_are_tagged(self):
        fake = RecordingSampler()
        p = make_div_p(fake, model_cls=_FakeZImage)
        run_div_pass(p, self.sigmas)
        self.assertEqual(fake.calls[0]["cond_scale"], 4.5)
        self.assertTrue(p.extra_generation_params["SPEED"].endswith(") experimental"))

    def test_two_coarse_segments_take_their_own_slices(self):
        fake = RecordingSampler()
        p = make_div_p(fake)
        seen = []
        run_div_pass(p, self.sigmas, seen=seen, scales="0.25,0.5,1.0",
                     manual_sigmas="0.95,0.9")
        self.assertEqual(len(fake.calls), 3)
        first, second, fine = fake.calls
        hi_idx = next(i for i, v in enumerate(self.sigmas) if float(v) <= 0.95)
        hi_sigma = float(self.sigmas[hi_idx])
        self.assertAlmostEqual(first["sigmas"][-1], hi_sigma, places=5)
        # the second segment starts at SPEED's aligned t_tilde, then follows the
        # reference strictly between the two transition sigmas
        self.assertGreater(second["sigmas"][0], hi_sigma)
        inner = [r for r in self.ref if self.t_sigma + 1e-4 < r < hi_sigma - 1e-4]
        self.assertEqual(len(second["sigmas"]), len(inner) + 2)
        for a, b in zip(second["sigmas"][1:-1], inner):
            self.assertAlmostEqual(a, b, places=5)
        self.assertAlmostEqual(second["sigmas"][-1], self.t_sigma, places=5)
        for call in (first, second):
            self.assertEqual(call["cond_scale"], 4.5)
            self.assertTrue(all(b < a for a, b in zip(call["sigmas"], call["sigmas"][1:])))
        self.assertIs(fine["extra_args"], ORIGINAL_ARGS)
        self.assertEqual(seen, list(range(12)))


class TestDiversityStandDowns(ForgeScriptTestCase):
    def setUp(self):
        super().setUp()
        self.sigmas = flow_schedule(12)

    def _assert_plain_speed(self, fake):
        for call in fake.calls:
            self.assertIs(call["extra_args"], ORIGINAL_ARGS)

    def test_feature_off(self):
        fake = RecordingSampler()
        p = make_div_p(fake)
        run_div_pass(p, self.sigmas, diversity_enabled=False)
        self.assertEqual(len(fake.calls), 2)
        self._assert_plain_speed(fake)
        self.assertEqual(p.encodes, [])
        self.assertEqual(p.reference_calls, [])
        self.assertNotIn("diversity", p.extra_generation_params["SPEED"])

    def test_speed_off(self):
        fake = RecordingSampler()
        p = make_div_p(fake)
        run_hooks(p, enabled=False, **DIV_UI)
        self.assertIs(p.sampler.func, fake)
        self.assertIsNone(speed_forge.SpeedForgeScript.diversity)
        self.assertEqual(p.encodes, [])

    def test_other_model_class(self):
        fake = RecordingSampler()
        p = make_div_p(fake)
        p.sd_model = SimpleNamespace(is_inpaint=False)
        run_div_pass(p, self.sigmas)
        self._assert_plain_speed(fake)
        self.assertIsNone(speed_forge.SpeedForgeScript.diversity)

    def test_empty_pattern(self):
        fake = RecordingSampler()
        run_div_pass(make_div_p(fake), self.sigmas, diversity_pattern=" , ")
        self._assert_plain_speed(fake)

    def test_hires_pass(self):
        fake = RecordingSampler()
        p = make_div_p(fake, is_hr_pass=True)
        run_div_pass(p, self.sigmas, apply_hr=True)
        self.assertTrue(getattr(p.sampler.func, "_speed_wrapped", False))
        self._assert_plain_speed(fake)
        self.assertEqual(p.encodes, [])

    def test_img2img_start_below_transition(self):
        fake = RecordingSampler()
        p = make_div_p(fake)
        start = next(i for i, v in enumerate(self.sigmas) if float(v) < 0.85)
        run_div_pass(p, self.sigmas[start:])
        self.assertEqual(len(fake.calls), 1)
        self._assert_plain_speed(fake)
        self.assertNotIn("diversity", p.extra_generation_params["SPEED"])

    def test_img2img_start_off_the_reference_keeps_own_steps(self):
        # starts above the transition but not on the 28-step reference: the CFG
        # stage still runs, the density stays the run's own
        fake = RecordingSampler()
        p = make_div_p(fake)
        sched = self.sigmas[1:]
        run_div_pass(p, sched)
        coarse = fake.calls[0]
        t_idx = next(i for i, v in enumerate(sched) if float(v) <= 0.9)
        self.assertEqual(coarse["steps"], t_idx)
        self.assertEqual(coarse["cond_scale"], 4.5)
        self.assertIs(fake.calls[1]["extra_args"], ORIGINAL_ARGS)

    def test_no_transition(self):
        fake = RecordingSampler()
        p = make_div_p(fake)
        run_div_pass(p, self.sigmas, manual_sigmas="0.0001")
        self.assertEqual(len(fake.calls), 1)
        self._assert_plain_speed(fake)
        self.assertNotIn("diversity", p.extra_generation_params["SPEED"])

    def test_guidance_override_logged(self):
        sys.modules["modules.shared"].opts.skip_early_cond = 0.2
        fake = RecordingSampler()
        out = []
        orig = speed_forge._log
        speed_forge._log = out.append
        try:
            run_div_pass(make_div_p(fake), self.sigmas)
        finally:
            speed_forge._log = orig
        self.assertTrue(any("Skip Early CFG" in m for m in out), out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
