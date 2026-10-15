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
_registered = {"before_ui": [], "after_component": [], "app_started": []}


def _install_stubs():
    class _Script:
        pass

    scripts_stub = types.ModuleType("modules.scripts")
    scripts_stub.Script = _Script
    scripts_stub.AlwaysVisible = object()
    scripts_stub.basedir = lambda: str(_ROOT)
    scripts_stub.scripts_data = []

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
    shared_stub.sd_model = None

    callbacks_stub = types.ModuleType("modules.script_callbacks")
    callbacks_stub.on_before_ui = _registered["before_ui"].append
    callbacks_stub.on_after_component = _registered["after_component"].append
    callbacks_stub.on_app_started = _registered["app_started"].append

    modules_stub = types.ModuleType("modules")
    modules_stub.script_callbacks = callbacks_stub
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
    sys.modules["modules.script_callbacks"] = callbacks_stub
    sys.modules["modules.shared"] = shared_stub
    sys.modules["k_diffusion"] = k_diffusion_stub
    sys.modules["k_diffusion.sampling"] = kds_stub

    networks_stub = types.ModuleType("networks")
    networks_stub.load_lora_for_models = _fake_load_lora_for_models
    sys.modules["networks"] = networks_stub
    return kds_stub


# --- fake Forge LoRA loader (extensions-builtin/sd_forge_lora/networks.py) ----

LOADER_CALLS = []


def _default_payload(strength):
    # [strength, adapter, strength_model, offset, function], a mutable list as
    # on current Forge Neo (backend/patcher/base.py add_patches)
    return [strength, "adapter", 1.0, None, None]


def _default_baked(strength, adapter):
    # baked payloads are tuples in patcher.patches[key]
    return (strength, adapter, 1.0, None, None)


_PAYLOAD = {"factory": _default_payload, "baked": _default_baked}

# adapter objects of the Turbo file's baked payloads, from the latest load
_TURBO_ADAPTERS = set()
_UUIDS = iter(range(1, 10 ** 9))

# the original (unpatched) value of every fake weight
ORIG = 0.0


class FakeOnlinePatch:
    def __init__(self, name, key, payload):
        self.name = name
        self.key = key
        self.patch = [payload]

    def __call__(self, weight):
        return weight


class FakeNet:
    """The nn.Module side: one weight value per key, shared by every patcher."""

    def __init__(self):
        self.weights = {}
        self.current_uuid = None


class FakePatcher:
    """ModelPatcher's clone semantics (backend/patcher/base.py): the wrapper
    dict and per-key patch lists are copied, the model and the backup dict are
    shared, and the patches uuid carries over."""

    def __init__(self, model=None):
        self.model = model if model is not None else FakeNet()
        self.weight_wrapper_patches = {}
        self.patches = {}
        self.backup = {}
        self.patches_uuid = 0

    def clone(self):
        n = FakePatcher(self.model)
        n.weight_wrapper_patches = self.weight_wrapper_patches.copy()
        n.patches = {k: v[:] for k, v in self.patches.items()}
        n.backup = self.backup
        n.patches_uuid = self.patches_uuid
        return n


def fake_model_load(unet):
    """partially_load: a changed uuid restores every backed-up weight and
    re-merges from the patches dict; an unchanged one leaves weights as they
    are (already merged)."""
    net = unet.model
    if net.current_uuid == unet.patches_uuid:
        return
    for key, bk in list(unet.backup.items()):
        net.weights[key] = bk.weight
    unet.backup.clear()
    for key, plist in unet.patches.items():
        unet.backup[key] = SimpleNamespace(weight=net.weights.get(key, ORIG), inplace_update=False)
        net.weights[key] = unet.backup[key].weight + sum(p[0] for p in plist)
    net.current_uuid = unet.patches_uuid


REBAKES = []
_REBAKE = {"fail": 0}


def fake_rebake_keys(patcher, keys):
    """speed_forge._rebake_keys on the fake weights: backup plus every
    non-zero payload in the patcher's own lists."""
    REBAKES.append(list(keys))
    if _REBAKE["fail"]:
        _REBAKE["fail"] -= 1
        raise RuntimeError("rebake failed")
    done = 0
    for key in keys:
        bk = patcher.backup.get(key)
        if bk is None:
            continue
        patcher.model.weights[key] = bk.weight + sum(
            p[0] for p in patcher.patches.get(key, []) if p[0] != 0.0)
        done += 1
    return done


def _fake_load_lora_for_models(model, clip, lora, strength_model, strength_clip,
                               filename="default", online_mode=False):
    LOADER_CALLS.append(dict(filename=filename, strength_model=strength_model,
                             strength_clip=strength_clip, online_mode=online_mode,
                             model=model, clip=clip, lora=lora))
    new = model.clone()
    for key in lora:
        if online_mode:
            obj = FakeOnlinePatch(filename, key, _PAYLOAD["factory"](strength_model))
            new.weight_wrapper_patches[key] = new.weight_wrapper_patches.get(key, []) + [obj]
        else:
            adapter = object()
            if filename == TURBO:
                _TURBO_ADAPTERS.add(adapter)
            plist = new.patches.pop(key, [])
            plist.append(_PAYLOAD["baked"](strength_model, adapter))
            new.patches[key] = plist
    new.patches_uuid = next(_UUIDS)
    return new, clip


_kds_stub = _install_stubs()

_spec = _iu.spec_from_file_location(
    "speed_forge_under_test", str(_ROOT / "scripts" / "speed_forge.py"))
speed_forge = _iu.module_from_spec(_spec)
sys.modules["speed_forge_under_test"] = speed_forge
_spec.loader.exec_module(speed_forge)

speed_core = speed_forge.speed_core
cfg_unlock = speed_forge.cfg_unlock


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
            "weights": dict(_SAMPLING["unet"].model.weights) if _SAMPLING["unet"] else {},
            "baked_payloads": _baked_turbo_payloads(_SAMPLING["unet"]),
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
                   diversity_cfg=4.5, diversity_pattern="turbo", diversity_bake=True)


def run_before(script, p, enabled=True, seeds=(123,), **overrides):
    """Drive before_process_batch with the UI values in Forge's order."""
    ui = {**UI_DEFAULTS, **overrides}
    script.before_process_batch(
        p, enabled, ui["transform"], ui["mode"], ui["model_preset"],
        ui["scales"], ui["adaptive_delta"], ui["delta"], ui["sigma_divisor"],
        ui["manual_sigmas"], ui["spectrum_A"], ui["spectrum_beta"], ui["seed"],
        ui["apply_hr"], ui["diversity_enabled"], ui["diversity_ref_steps"],
        ui["diversity_cfg"], ui["diversity_pattern"], ui["diversity_bake"],
        seeds=list(seeds), prompts=["test"], subseeds=[0])


def run_hooks(p, enabled=True, seeds=(123,), uc=None, loras=None, **overrides):
    """Drive before_process_batch [+ LoRA activation] + process_before_every_sampling."""
    script = speed_forge.SpeedForgeScript()
    run_before(script, p, enabled, seeds, **overrides)
    if loras is not None:
        fake_activate(p, loras)
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


class _FakeForgeModel:
    is_inpaint = False

    def __init__(self):
        self.original_unet = FakePatcher()
        self.forge_objects = SimpleNamespace(unet=self.original_unet, clip=None)
        self.forge_objects_after_applying_lora = SimpleNamespace(unet=self.original_unet)
        self.current_lora_hash = "[]"


class _FakeKrea2(_FakeForgeModel):
    pass


_FakeKrea2.__name__ = "Krea2"


class _FakeZImage(_FakeForgeModel):
    pass


_FakeZImage.__name__ = "ZImage"

TURBO = "/loras/krea2_turbo_lora_rank_64_bf16.safetensors"
STYLE = "/loras/style_detail.safetensors"
LORA_KEYS = {
    TURBO: ["blocks.0.attn.weight", "blocks.1.mlp.weight"],
    # shares a key with the Turbo LoRA: online and baked coexist per key
    STYLE: ["blocks.0.attn.weight", "blocks.5.mlp.weight"],
}
DEFAULT_LORAS = ((TURBO, 0.6), (STYLE, 0.8))


def fake_activate(p, loras, online=False):
    """networks.load_networks: hash-cached, resets to the original unet, then
    loads each LoRA through the module attribute (so interception applies)."""
    m = p.sd_model
    if not hasattr(m, "forge_objects"):
        return
    h = str([[f, s, s, online] for f, s in loras])
    if m.current_lora_hash == h:
        return
    m.current_lora_hash = h
    unet = m.original_unet
    _TURBO_ADAPTERS.clear()
    for f, s in loras:
        unet, _ = sys.modules["networks"].load_lora_for_models(
            unet, None, LORA_KEYS[f], s, s, filename=f, online_mode=online)
    m.forge_objects.unet = unet
    m.forge_objects_after_applying_lora = SimpleNamespace(unet=unet)
    _TURBO_OBJECTS[:] = [o for objs in unet.weight_wrapper_patches.values()
                         for o in objs if o.name == TURBO]

# Turbo LoRA patch objects whose strength RecordingSampler records per call.
_TURBO_OBJECTS = []
# the unet loaded for sampling in the current fake pass
_SAMPLING = {"unet": None}


def _baked_turbo_payloads(unet):
    if unet is None:
        return []
    return [float(p[0]) for key in sorted(unet.patches) for p in unet.patches[key]
            if p[1] in _TURBO_ADAPTERS]


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


def sample_pass(p, sigmas, model=None, seen=None):
    """sampler.sample: load the unet about to sample, then run the (wrapped)
    sampler function."""
    unet = getattr(getattr(p.sd_model, "forge_objects", None), "unet", None)
    if isinstance(unet, FakePatcher):
        fake_model_load(unet)
        _SAMPLING["unet"] = unet
    p.sampler.func(model, torch.randn(1, 4, 32, 32), sigmas=sigmas,
                   extra_args=ORIGINAL_ARGS, disable=True,
                   callback=None if seen is None else (lambda d: seen.append(d["i"])))


def run_div_pass(p, sigmas, model=None, seen=None, loras=DEFAULT_LORAS, uc=None,
                 after_hooks=None, **ui):
    """Hooks, LoRA activation and one wrapped sampling pass; returns the script."""
    script = run_hooks(p, loras=loras, uc=uc, **{**DIV_UI, **ui})
    if after_hooks is not None:
        after_hooks(p)
    sample_pass(p, sigmas, model=model, seen=seen)
    return script


def capture_log():
    """Swap speed_forge._log for a list collector; returns (list, restore)."""
    out = []
    orig = speed_forge._log
    speed_forge._log = out.append
    return out, lambda: setattr(speed_forge, "_log", orig)


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
        cls.turbo_route = None
        cls.turbo_membership = None
        cls.turbo_cycle_open = False
        cls.turbo_entries = []
        cls.turbo_blockers = []
        _kds_stub.torch = torch
        sys.modules["modules.shared"].opts.skip_early_cond = 0.0
        sys.modules["modules.scripts"].scripts_data = []
        cls.turbo_baked = []
        cls.turbo_baked_patterns = None
        cls.turbo_unbaked = None
        cls.turbo_pinned = False
        cls.turbo_pin_off = None
        LOADER_CALLS.clear()
        REBAKES.clear()
        _REBAKE["fail"] = 0
        _TURBO_OBJECTS.clear()
        _TURBO_ADAPTERS.clear()
        _SAMPLING["unet"] = None
        _PAYLOAD["factory"] = _default_payload
        _PAYLOAD["baked"] = _default_baked
        speed_forge._rebake_keys = fake_rebake_keys


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
        run_div_pass(p, self.sigmas, uc="FORGE_UC")
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
            " diversity(ref=28, cfg=4.5, lora=krea2_turbo_lora_rank_64_bf16:0.6, "
            f"coarse={fake.calls[0]['steps']})"), info)

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
        out, restore = capture_log()
        try:
            run_div_pass(make_div_p(fake), self.sigmas)
        finally:
            restore()
        self.assertTrue(any("Skip Early CFG" in m for m in out), out)



class TestTurboLoRA(ForgeScriptTestCase):
    """The on-the-fly route (Bake unchecked)."""

    def setUp(self):
        super().setUp()
        self.sigmas = flow_schedule(12)
        DIV_UI["diversity_bake"] = False

    def tearDown(self):
        DIV_UI.pop("diversity_bake", None)

    def _calls_for(self, filename):
        return [c for c in LOADER_CALLS if c["filename"] == filename]

    def _assert_stood_down(self, fake):
        for call in fake.calls:
            self.assertIs(call["extra_args"], ORIGINAL_ARGS)
            self.assertTrue(all(v == 0.6 for v in call["turbo_strengths"]))

    def test_only_the_turbo_file_is_routed_online(self):
        p = make_div_p(RecordingSampler())
        run_div_pass(p, self.sigmas)
        (turbo,), (style,) = self._calls_for(TURBO), self._calls_for(STYLE)
        self.assertTrue(turbo["online_mode"])
        self.assertEqual(turbo["strength_model"], 0.6)
        # the other LoRA reaches the stock loader with its original arguments
        self.assertFalse(style["online_mode"])
        self.assertEqual((style["strength_model"], style["strength_clip"]), (0.8, 0.8))
        self.assertIs(style["lora"], LORA_KEYS[STYLE])
        unet = p.sd_model.forge_objects.unet
        self.assertEqual(sorted(unet.patches), sorted(LORA_KEYS[STYLE]))
        self.assertEqual(sorted(unet.weight_wrapper_patches), sorted(LORA_KEYS[TURBO]))
        self.assertEqual(len(speed_forge.SpeedForgeScript.turbo_entries), 2)

    def test_strength_zero_for_coarse_base_for_fine(self):
        fake = RecordingSampler()
        run_div_pass(make_div_p(fake), self.sigmas)
        coarse, fine = fake.calls
        self.assertEqual(coarse["turbo_strengths"], [0.0, 0.0])
        self.assertEqual(fine["turbo_strengths"], [0.6, 0.6])
        self.assertEqual([o.patch[0][0] for o in _TURBO_OBJECTS], [0.6, 0.6])

    def test_parked_at_base_by_postprocess(self):
        fake = RecordingSampler()
        p = make_div_p(fake)
        script = run_div_pass(p, self.sigmas)
        for o in _TURBO_OBJECTS:
            o.patch[0][0] = 0.0   # e.g. a run that ended outside the wrapper
        script.postprocess(p, None)
        self.assertEqual([o.patch[0][0] for o in _TURBO_OBJECTS], [0.6, 0.6])

    def test_interrupted_coarse_segment_leaves_lora_whole(self):
        fake = RecordingSampler(raise_at_call=1)
        p = make_div_p(fake)
        with self.assertRaises(RuntimeError):
            run_div_pass(p, self.sigmas)
        self.assertEqual(fake.calls[0]["turbo_strengths"], [0.0, 0.0])
        self.assertEqual([o.patch[0][0] for o in _TURBO_OBJECTS], [0.6, 0.6])

    def test_tuple_payloads_are_rebuilt(self):
        _PAYLOAD["factory"] = lambda s: (s, "adapter", 1.0, None, None)
        fake = RecordingSampler()
        run_div_pass(make_div_p(fake), self.sigmas)
        self.assertEqual(fake.calls[0]["turbo_strengths"], [0.0, 0.0])
        self.assertEqual(fake.calls[1]["turbo_strengths"], [0.6, 0.6])

    def test_no_matching_lora_stands_down(self):
        fake = RecordingSampler()
        p = make_div_p(fake)
        out, restore = capture_log()
        try:
            run_div_pass(p, self.sigmas, loras=((STYLE, 0.8),))
        finally:
            restore()
        self._assert_stood_down(fake)
        self.assertTrue(any("no prompt LoRA matches the Turbo pattern 'turbo'" in m
                            for m in out), out)
        self.assertEqual(p.encodes, [])
        self.assertNotIn("diversity", p.extra_generation_params["SPEED"])
        self.assertFalse(self._calls_for(STYLE)[0]["online_mode"])

    def test_pattern_alternatives(self):
        fake = RecordingSampler()
        run_div_pass(make_div_p(fake), self.sigmas, diversity_pattern="lightning, STYLE")
        # the style file now matches, the turbo file does not
        self.assertTrue(self._calls_for(STYLE)[0]["online_mode"])
        self.assertFalse(self._calls_for(TURBO)[0]["online_mode"])
        self.assertEqual(fake.calls[0]["cond_scale"], 4.5)

    def test_drift_refused(self):
        _PAYLOAD["factory"] = lambda s: [s, "adapter", 1.0, None, None, True]
        fake = RecordingSampler()
        p = make_div_p(fake)
        out, restore = capture_log()
        try:
            run_div_pass(p, self.sigmas)
        finally:
            restore()
        self._assert_stood_down(fake)
        self.assertEqual(speed_forge.SpeedForgeScript.turbo_entries, [])
        self.assertTrue(any("refusing to drive its patches" in m and "arity" in m
                            for m in out), out)

    def test_outer_interceptor_already_routing_online(self):
        networks = sys.modules["networks"]
        inner = networks.load_lora_for_models

        def outer(model, clip, lora, sm, sc, filename="default", online_mode=False):
            # another extension installed after us, forcing on-the-fly loading
            return inner(model, clip, lora, sm, sc, filename,
                         True if "turbo" in filename else online_mode)

        networks.load_lora_for_models = outer
        try:
            fake = RecordingSampler()
            out, restore = capture_log()
            try:
                run_div_pass(make_div_p(fake), self.sigmas)
            finally:
                restore()
        finally:
            networks.load_lora_for_models = inner
        self._assert_stood_down(fake)
        self.assertEqual(speed_forge.SpeedForgeScript.turbo_entries, [])
        self.assertTrue(any("already loads it on the fly" in m for m in out), out)

    def test_global_on_the_fly_option_is_not_a_conflict(self):
        with _fake_backend_args(ref_latents=[]) as args:
            args.dynamic_args.online_lora = True
            fake = RecordingSampler()
            p = make_div_p(fake)
            script = speed_forge.SpeedForgeScript()
            run_before(script, p, **DIV_UI)
            fake_activate(p, DEFAULT_LORAS, online=True)
            script.process_before_every_sampling(p, uc=None)
            sample_pass(p, self.sigmas)
        self.assertEqual(fake.calls[0]["turbo_strengths"], [0.0, 0.0])
        self.assertEqual(fake.calls[1]["turbo_strengths"], [0.6, 0.6])

    def test_builtin_lora_control_owns_it(self):
        class LoRAControl:
            mapping = {TURBO: [(0.0, 0.0), (1.0, 1.0)]}

        sys.modules["modules.scripts"].scripts_data = [SimpleNamespace(script_class=LoRAControl)]
        fake = RecordingSampler()
        out, restore = capture_log()
        try:
            run_div_pass(make_div_p(fake), self.sigmas)
        finally:
            restore()
        self._assert_stood_down(fake)
        # passed through untouched, not routed by us
        self.assertFalse(self._calls_for(TURBO)[0]["online_mode"])
        self.assertTrue(any("LoRA Control Integrated" in m for m in out), out)

    def test_neo_loractl_owns_it(self):
        class NeoLoraCtlScript:
            enabled = True
            patterns = ["turbo"]
            filter_mode = "include"

        loractl_core = types.ModuleType("neo_loractl_core")

        def lora_is_scheduled(filename, patterns, mode):
            matched = any(p in filename.lower() for p in patterns)
            return matched if mode == "include" else not matched

        loractl_core.lora_is_scheduled = lora_is_scheduled
        sys.modules["neo_loractl_core"] = loractl_core
        sys.modules["modules.scripts"].scripts_data = [
            SimpleNamespace(script_class=NeoLoraCtlScript)]
        try:
            fake = RecordingSampler()
            run_div_pass(make_div_p(fake), self.sigmas)
            self._assert_stood_down(fake)
            # with its default exclude filter it leaves the Turbo LoRA alone
            NeoLoraCtlScript.filter_mode = "exclude"
            fake2 = RecordingSampler()
            run_div_pass(make_div_p(fake2), self.sigmas)
            self.assertEqual(fake2.calls[0]["turbo_strengths"], [0.0, 0.0])
        finally:
            sys.modules.pop("neo_loractl_core", None)

    def test_lora_hash_nulled_only_on_membership_change(self):
        p = make_div_p(RecordingSampler())
        m = p.sd_model

        def before(**ui):
            run_before(speed_forge.SpeedForgeScript(), p, **{**DIV_UI, **ui})
            return m.current_lora_hash

        self.assertEqual(before(diversity_enabled=False), "[]")      # off -> off
        self.assertIsNone(before())                                   # off -> on
        fake_activate(p, DEFAULT_LORAS)
        cached = m.current_lora_hash
        self.assertEqual(before(), cached)                            # on -> on
        self.assertEqual(before(diversity_cfg=6.0, diversity_ref_steps=40), cached)
        self.assertIsNone(before(diversity_pattern="turbo, lightning"))  # pattern
        fake_activate(p, DEFAULT_LORAS)
        self.assertIsNone(before(diversity_enabled=False))            # on -> off
        fake_activate(p, DEFAULT_LORAS)
        cached = m.current_lora_hash
        self.assertEqual(before(diversity_enabled=False), cached)     # off -> off
        self.assertEqual(before(enabled=False), cached)               # SPEED off too
        self.assertIsNone(before())                                   # back on
        fake_activate(p, DEFAULT_LORAS)
        cached = m.current_lora_hash
        self.assertIsNone(before(enabled=False))                      # SPEED off

    def test_unsupported_model_never_routes(self):
        p = make_div_p(RecordingSampler())
        p.sd_model.__class__ = type("Flux", (_FakeForgeModel,), {})
        run_div_pass(p, self.sigmas)
        self.assertFalse(self._calls_for(TURBO)[0]["online_mode"])
        self.assertEqual(p.sd_model.current_lora_hash.count("False"), 2)

    def test_toggling_off_reloads_the_turbo_lora_baked(self):
        fake = RecordingSampler()
        p = make_div_p(fake)
        run_div_pass(p, self.sigmas)
        LOADER_CALLS.clear()
        run_div_pass(p, self.sigmas, diversity_enabled=False)
        self.assertFalse(self._calls_for(TURBO)[0]["online_mode"])
        self.assertEqual(p.sd_model.forge_objects.unet.weight_wrapper_patches, {})

    def test_cached_loader_keeps_adoption_across_runs(self):
        p = make_div_p(RecordingSampler())
        run_div_pass(p, self.sigmas).postprocess(p, None)
        LOADER_CALLS.clear()
        fake = RecordingSampler()
        p.sampler.func = fake
        run_div_pass(p, self.sigmas)
        self.assertEqual(LOADER_CALLS, [])          # the stock loader's cache hit
        self.assertEqual(fake.calls[0]["turbo_strengths"], [0.0, 0.0])
        self.assertEqual(fake.calls[1]["turbo_strengths"], [0.6, 0.6])

    def test_turbo_removed_from_prompt_stands_down(self):
        p = make_div_p(RecordingSampler())
        run_div_pass(p, self.sigmas).postprocess(p, None)
        fake = RecordingSampler()
        p.sampler.func = fake
        _TURBO_OBJECTS.clear()
        run_div_pass(p, self.sigmas, loras=((STYLE, 0.8),))
        self._assert_stood_down(fake)
        self.assertEqual(speed_forge.SpeedForgeScript.turbo_entries, [])

    def test_stale_entries_from_an_earlier_load_are_dropped(self):
        # hires prompts can re-run activate inside setup_conds; only the
        # objects in the unet about to sample may be driven
        p = make_div_p(RecordingSampler())
        script = speed_forge.SpeedForgeScript()
        run_before(script, p, **DIV_UI)
        fake_activate(p, DEFAULT_LORAS)
        stale = list(_TURBO_OBJECTS)
        fake_activate(p, ((TURBO, 0.5),))
        fresh = list(_TURBO_OBJECTS)
        self.assertEqual(len(speed_forge.SpeedForgeScript.turbo_entries), 4)
        script.process_before_every_sampling(p, uc=None)
        sample_pass(p, self.sigmas)
        self.assertEqual([o.patch[0][0] for o in stale], [0.6, 0.6])
        self.assertEqual([o.patch[0][0] for o in fresh], [0.5, 0.5])
        self.assertEqual(len(speed_forge.SpeedForgeScript.turbo_entries), 2)
        self.assertTrue(p.extra_generation_params["SPEED"].endswith(
            "lora=krea2_turbo_lora_rank_64_bf16:0.5, switch=online, coarse=10)"))



class TestTurboLoRABaked(ForgeScriptTestCase):
    """The default route: Forge's own baked load, two weight rewrites per pass."""

    TURBO_ONLY = "blocks.1.mlp.weight"
    SHARED = "blocks.0.attn.weight"     # turbo 0.6 + style 0.8

    def setUp(self):
        super().setUp()
        self.sigmas = flow_schedule(12)

    def _at(self, call):
        return (call["weights"].get(self.TURBO_ONLY), call["weights"].get(self.SHARED))

    def test_the_load_is_forges_own(self):
        p = make_div_p(RecordingSampler())
        run_div_pass(p, self.sigmas)
        self.assertEqual([c["online_mode"] for c in LOADER_CALLS], [False, False])
        unet = p.sd_model.forge_objects.unet
        self.assertEqual(unet.weight_wrapper_patches, {})
        self.assertEqual(len(speed_forge.SpeedForgeScript.turbo_baked), 2)
        self.assertEqual(speed_forge.SpeedForgeScript.turbo_entries, [])

    def test_off_for_coarse_on_for_fine(self):
        fake = RecordingSampler()
        p = make_div_p(fake)
        run_div_pass(p, self.sigmas)
        coarse, fine = fake.calls
        self.assertEqual(self._at(coarse), (0.0, 0.8))
        self.assertEqual(self._at(fine), (0.6, 0.6 + 0.8))
        # the payloads the lowvram path would read follow the switch too
        self.assertEqual(coarse["baked_payloads"], [0.0, 0.0])
        self.assertEqual(fine["baked_payloads"], [0.6, 0.6])
        self.assertEqual(coarse["cond_scale"], 4.5)
        self.assertEqual(len(REBAKES), 2)
        self.assertEqual(sorted(REBAKES[0]), sorted(LORA_KEYS[TURBO]))
        # back at rest: Forge's own baked state, and no full repatch requested
        unet = p.sd_model.forge_objects.unet
        self.assertEqual((unet.model.weights[self.TURBO_ONLY], unet.model.weights[self.SHARED]),
                         (0.6, 1.4))
        self.assertEqual(unet.model.current_uuid, unet.patches_uuid)
        self.assertIsNone(speed_forge.SpeedForgeScript.turbo_unbaked)

    def test_infotext_has_no_switch_tag(self):
        p = make_div_p(RecordingSampler())
        run_div_pass(p, self.sigmas)
        self.assertTrue(p.extra_generation_params["SPEED"].endswith(
            "lora=krea2_turbo_lora_rank_64_bf16:0.6, coarse=10)"))

    def test_the_unet_that_samples_is_the_one_switched(self):
        # another script clones the unet after our hook; the clone shares the
        # model and backups but has its own patch lists
        fake = RecordingSampler()
        p = make_div_p(fake)
        holder = {}

        def clone_unet(p):
            holder["cached"] = p.sd_model.forge_objects.unet
            p.sd_model.forge_objects.unet = holder["cached"].clone()

        run_div_pass(p, self.sigmas, after_hooks=clone_unet)
        self.assertEqual(self._at(fake.calls[0]), (0.0, 0.8))
        self.assertEqual(fake.calls[0]["baked_payloads"], [0.0, 0.0])
        cached = holder["cached"]
        self.assertEqual(sorted(float(p_[0]) for k in cached.patches for p_ in cached.patches[k]),
                         [0.6, 0.6, 0.8, 0.8])

    def test_interrupted_coarse_segment_rebakes(self):
        fake = RecordingSampler(raise_at_call=1)
        p = make_div_p(fake)
        with self.assertRaises(RuntimeError):
            run_div_pass(p, self.sigmas)
        self.assertEqual(self._at(fake.calls[0]), (0.0, 0.8))
        w = p.sd_model.forge_objects.unet.model.weights
        self.assertEqual((w[self.TURBO_ONLY], w[self.SHARED]), (0.6, 1.4))
        self.assertIsNone(speed_forge.SpeedForgeScript.turbo_unbaked)

    def test_postprocess_rebakes_a_pass_left_off(self):
        p = make_div_p(RecordingSampler())
        script = run_div_pass(p, self.sigmas)
        cls = speed_forge.SpeedForgeScript
        unet = p.sd_model.forge_objects.unet
        speed_forge._switch_baked(unet, cls.turbo_baked, 0.0)
        cls.turbo_unbaked = (unet, cls.turbo_baked)
        self.assertEqual(unet.model.weights[self.TURBO_ONLY], 0.0)
        script.postprocess(p, None)
        self.assertEqual(unet.model.weights[self.TURBO_ONLY], 0.6)
        self.assertIsNone(cls.turbo_unbaked)

    def test_failed_rewrite_restores_and_forces_a_full_repatch(self):
        _REBAKE["fail"] = 1
        fake = RecordingSampler()
        p = make_div_p(fake)
        with self.assertRaises(RuntimeError):
            run_div_pass(p, self.sigmas)
        self.assertEqual(fake.calls, [])          # failed before the coarse call
        unet = p.sd_model.forge_objects.unet
        self.assertEqual(_baked_turbo_payloads(unet), [0.6, 0.6])
        self.assertNotEqual(unet.model.current_uuid, unet.patches_uuid)
        fake_model_load(unet)                     # the next load rebuilds
        self.assertEqual(unet.model.weights[self.TURBO_ONLY], 0.6)

    def test_cached_load_keeps_adoption_and_still_switches(self):
        p = make_div_p(RecordingSampler())
        run_div_pass(p, self.sigmas).postprocess(p, None)
        LOADER_CALLS.clear()
        fake = RecordingSampler()
        p.sampler.func = fake
        run_div_pass(p, self.sigmas)
        self.assertEqual(LOADER_CALLS, [])
        self.assertEqual(self._at(fake.calls[0]), (0.0, 0.8))
        self.assertEqual(self._at(fake.calls[1]), (0.6, 1.4))

    def test_lora_hash_only_reloads_when_needed(self):
        p = make_div_p(RecordingSampler())
        m = p.sd_model

        def before(**ui):
            run_before(speed_forge.SpeedForgeScript(), p, **{**DIV_UI, **ui})
            return m.current_lora_hash

        self.assertIsNone(before())                       # first enable: must see the load
        fake_activate(p, DEFAULT_LORAS)
        cached = m.current_lora_hash
        self.assertEqual(before(), cached)                # unchanged
        self.assertEqual(before(diversity_enabled=False), cached)   # off keeps Forge's load
        self.assertEqual(before(), cached)                # back on, payloads still loaded
        self.assertEqual(before(enabled=False), cached)   # SPEED off
        self.assertEqual(before(), cached)
        self.assertIsNone(before(diversity_pattern="lightning"))     # new pattern
        fake_activate(p, DEFAULT_LORAS)
        self.assertIsNone(before())                       # back to turbo: not adopted now
        fake_activate(p, DEFAULT_LORAS)
        cached = m.current_lora_hash
        self.assertIsNone(before(diversity_bake=False))   # to on-the-fly: reload
        fake_activate(p, DEFAULT_LORAS)
        self.assertIsNone(before())                       # and back: reload
        fake_activate(p, DEFAULT_LORAS)
        cached = m.current_lora_hash
        # a reload while off drops the adoption; switching on reloads again
        self.assertEqual(before(diversity_enabled=False), cached)
        m.current_lora_hash = "something else"
        fake_activate(p, DEFAULT_LORAS)
        self.assertIsNone(before())

    def test_no_matching_lora_stands_down(self):
        fake = RecordingSampler()
        p = make_div_p(fake)
        run_div_pass(p, self.sigmas, loras=((STYLE, 0.8),))
        for call in fake.calls:
            self.assertIs(call["extra_args"], ORIGINAL_ARGS)
        self.assertEqual(REBAKES, [])

    def test_baked_drift_refused(self):
        _PAYLOAD["baked"] = lambda s_, a: (s_, a, 1.0, None, None, True)
        fake = RecordingSampler()
        p = make_div_p(fake)
        out, restore = capture_log()
        try:
            run_div_pass(p, self.sigmas, loras=((TURBO, 0.6),))
        finally:
            restore()
        self.assertEqual(speed_forge.SpeedForgeScript.turbo_baked, [])
        self.assertEqual(REBAKES, [])
        self.assertTrue(any("refusing to drive its patches" in m for m in out), out)

    def test_global_on_the_fly_option_drives_it_online(self):
        with _fake_backend_args(ref_latents=[]) as args:
            args.dynamic_args.online_lora = True
            fake = RecordingSampler()
            p = make_div_p(fake)
            script = speed_forge.SpeedForgeScript()
            run_before(script, p, **DIV_UI)
            fake_activate(p, DEFAULT_LORAS, online=True)
            script.process_before_every_sampling(p, uc=None)
            sample_pass(p, self.sigmas)
        self.assertEqual(fake.calls[0]["turbo_strengths"], [0.0, 0.0])
        self.assertEqual(REBAKES, [])
        self.assertIn("switch=online", p.extra_generation_params["SPEED"])

    def test_flip_times_are_logged(self):
        out, restore = capture_log()
        try:
            run_div_pass(make_div_p(RecordingSampler()), self.sigmas)
        finally:
            restore()
        flips = [m for m in out if "Turbo LoRA off" in m or "Turbo LoRA back on" in m]
        self.assertEqual(len(flips), 2, out)
        self.assertIn("2 of 2 weights rewritten", flips[0])



Dimension = __import__("collections").namedtuple("Dimension", ["weight", "inplace_update"])


class _QuantTensor(torch.Tensor):
    """Stands in for a quantized tensor subclass, which must not be pinned."""


class TestPinBackups(ForgeScriptTestCase):
    def setUp(self):
        super().setUp()
        self.pinned = []
        self.saved = {name: getattr(speed_forge, name) for name in
                      ("_pin_available", "_pin_tensor", "_is_pinned", "_ram_total",
                       "_release_pinned", "_pin_backups")}
        speed_forge._pin_available = lambda: True

        def pin(t):
            copy = t.clone()
            self.pinned.append(copy)
            return copy

        speed_forge._pin_tensor = pin
        speed_forge._is_pinned = lambda t: any(t.data_ptr() == x.data_ptr() for x in self.pinned)
        speed_forge._ram_total = lambda: 64 * 2 ** 30
        self.sigmas = flow_schedule(12)

    def tearDown(self):
        for name, value in self.saved.items():
            setattr(speed_forge, name, value)

    def _patcher(self):
        patcher = FakePatcher()
        param = torch.nn.Parameter(torch.ones(4, 4), requires_grad=False)
        patcher.backup = {
            "a.weight": Dimension(torch.arange(16, dtype=torch.int8).reshape(4, 4), False),
            "b.weight": Dimension(param, True),
            "other.weight": Dimension(torch.zeros(2), False),
            "quant.weight": Dimension(torch.zeros(3).as_subclass(_QuantTensor), False),
        }
        return patcher

    def test_pins_the_given_plain_cpu_backups_once(self):
        patcher = self._patcher()
        before = dict(patcher.backup)
        n, size, _ = speed_forge._pin_backups(
            patcher, ["a.weight", "b.weight", "quant.weight", "missing.weight"])
        self.assertEqual(n, 2)
        self.assertEqual(size, 16 * 1 + 16 * 4)
        a, b = patcher.backup["a.weight"], patcher.backup["b.weight"]
        self.assertIs(a.weight, self.pinned[0])
        self.assertTrue(torch.equal(a.weight, before["a.weight"].weight))
        self.assertEqual(a.weight.dtype, torch.int8)
        self.assertFalse(a.inplace_update)
        self.assertIsInstance(b.weight, torch.nn.Parameter)
        self.assertTrue(b.inplace_update)
        self.assertIs(patcher.backup["other.weight"], before["other.weight"])
        self.assertIs(patcher.backup["quant.weight"], before["quant.weight"])
        # clones share the backup dict, so they see the pinned copies
        self.assertIs(patcher.clone().backup["a.weight"], a)
        self.assertEqual(speed_forge._pin_backups(patcher, ["a.weight", "b.weight"]),
                         (0, 0, 0.0))

    def test_budget_and_failure_give_up_for_the_session(self):
        cls = speed_forge.SpeedForgeScript
        baked = [("a.weight", object(), 0.6, "turbo"), ("b.weight", object(), 0.6, "turbo")]
        out, restore = capture_log()
        try:
            speed_forge._ram_total = lambda: 100      # 40% of 100 bytes
            patcher = self._patcher()
            cls._pin_turbo_backups(patcher, baked)
            self.assertEqual(cls.turbo_pin_off, "budget")
            self.assertEqual(self.pinned, [])
            speed_forge._ram_total = lambda: 64 * 2 ** 30
            cls._pin_turbo_backups(patcher, baked)    # not retried
            self.assertEqual(self.pinned, [])

            cls.turbo_pin_off = None

            def boom(t):
                raise RuntimeError("cudaHostAlloc failed")

            speed_forge._pin_tensor = boom
            cls._pin_turbo_backups(patcher, baked)
            self.assertEqual(cls.turbo_pin_off, "cudaHostAlloc failed")
            self.assertFalse(cls.turbo_pinned)
        finally:
            restore()
        self.assertEqual(sum("not pinning them" in m for m in out), 1, out)
        self.assertEqual(sum("could not pin" in m for m in out), 1, out)

    def test_pinned_before_the_first_off_switch_only(self):
        order = []
        real_rebake = speed_forge._rebake_keys

        def spy_pin(patcher, keys):
            order.append(("pin", sorted(keys)))
            return 2, 1024, 0.01

        def spy_rebake(patcher, keys):
            order.append(("rebake",))
            return real_rebake(patcher, keys)

        speed_forge._pin_backups = spy_pin
        speed_forge._rebake_keys = spy_rebake
        out, restore = capture_log()
        try:
            run_div_pass(make_div_p(RecordingSampler()), self.sigmas)
        finally:
            restore()
        self.assertEqual(order, [("pin", sorted(LORA_KEYS[TURBO])), ("rebake",), ("rebake",)])
        self.assertTrue(speed_forge.SpeedForgeScript.turbo_pinned)
        self.assertTrue(any("pinned 2 Turbo weight backups" in m for m in out), out)

    def test_cached_pinned_memory_released_once_the_adoption_is_gone(self):
        released = []
        speed_forge._release_pinned = lambda: released.append(True)
        speed_forge._pin_backups = lambda patcher, keys: (2, 1024, 0.01)
        p = make_div_p(RecordingSampler())
        run_div_pass(p, self.sigmas).postprocess(p, None)
        self.assertTrue(speed_forge.SpeedForgeScript.turbo_pinned)
        # feature off, LoRA still loaded: its backups are still in use
        run_before(speed_forge.SpeedForgeScript(), p, **{**DIV_UI, "diversity_enabled": False})
        self.assertEqual(released, [])
        # the prompt changes and Forge reloads without the Turbo LoRA
        fake_activate(p, ((STYLE, 0.8),))
        run_before(speed_forge.SpeedForgeScript(), p, **{**DIV_UI, "diversity_enabled": False})
        self.assertEqual(released, [True])
        self.assertFalse(speed_forge.SpeedForgeScript.turbo_pinned)
        run_before(speed_forge.SpeedForgeScript(), p, **{**DIV_UI, "diversity_enabled": False})
        self.assertEqual(released, [True])



# =============================================================================
# Negative prompt unlock (scripts/lib_speed_forge/cfg_unlock.py)
# =============================================================================

def forge_use_cfg(val):
    """modules/ui.py's rule, with a recognizable return value."""
    return "skip" if val is None else ("forge", val > 1.0)


class FakeNag:
    """Neo-Krea2-NAG's lib_krea2_nag/cfg_unlock.py, reduced to its substitution
    logic: install at before_ui (recognizing only its own tag), overwrite at
    img2img_cfg_scale."""

    def __init__(self, ui):
        self.ui = ui
        self.original = None
        self.enabled = {"txt2img": False, "img2img": False}

    def make(self, tab):
        def use_cfg(val):
            if self.enabled[tab]:
                return ("update", {"interactive": True})
            return self.original(val)
        use_cfg._krea2_nag_tab = tab
        return use_cfg

    def install(self):
        current = self.ui.use_cfg
        if getattr(current, "_krea2_nag_tab", None) is None:
            self.original = current
        self.ui.use_cfg = self.make("txt2img")

    def swap(self, component, **kw):
        if self.original is not None and component.elem_id == "img2img_cfg_scale":
            self.ui.use_cfg = self.make("img2img")


class TestCfgUnlock(unittest.TestCase):
    def setUp(self):
        self.ui = types.ModuleType("modules.ui")
        self.ui.use_cfg = forge_use_cfg
        sys.modules["modules.ui"] = self.ui
        sys.modules["modules"].ui = self.ui
        self.saved_gr = cfg_unlock.gr
        cfg_unlock.gr = SimpleNamespace(update=lambda **kw: ("update", kw), skip=lambda: "skip")
        cfg_unlock._enabled.update(txt2img=False, img2img=False)
        cfg_unlock._installed = False
        cfg_unlock._base_use_cfg = None

    def tearDown(self):
        cfg_unlock.gr = self.saved_gr
        sys.modules.pop("modules.ui", None)
        if hasattr(sys.modules["modules"], "ui"):
            del sys.modules["modules"].ui

    def build(self, before_ui, after_component):
        """modules/ui.py create_ui: which use_cfg each .change() binding captures."""
        for cb in before_ui:
            cb()
        bindings = {}

        def component(elem_id):
            for cb in after_component:
                cb(SimpleNamespace(elem_id=elem_id))

        for elem_id in ("txt2img_prompt", "txt2img_neg_prompt", "txt2img_cfg_scale"):
            component(elem_id)
        bindings["txt2img"] = self.ui.use_cfg
        for elem_id in ("txt2img_hr_cfg", "hires_neg_prompt"):
            component(elem_id)
        bindings["hires"] = self.ui.use_cfg
        for elem_id in ("img2img_neg_prompt", "img2img_cfg_scale", "img2img_image_cfg_scale"):
            component(elem_id)
        bindings["img2img"] = self.ui.use_cfg
        return bindings

    @staticmethod
    def unlocked(result):
        return result == ("update", {"interactive": True}) or result == ("forge", True)

    def test_registered_with_forge_callbacks(self):
        self.assertIn(cfg_unlock._before_ui, _registered["before_ui"])
        self.assertIn(cfg_unlock._on_component, _registered["after_component"])

    def test_alone_each_tab_has_its_own_flag(self):
        b = self.build([cfg_unlock._before_ui], [cfg_unlock._on_component])
        self.assertEqual(b["txt2img"](1.0), ("forge", False))
        self.assertEqual(b["txt2img"](4.0), ("forge", True))
        self.assertEqual(b["txt2img"](None), "skip")
        cfg_unlock.set_enabled("txt2img", True)
        self.assertTrue(self.unlocked(b["txt2img"](1.0)))
        self.assertEqual(b["img2img"](1.0), ("forge", False))
        cfg_unlock.set_enabled("img2img", True)
        cfg_unlock.set_enabled("txt2img", False)
        self.assertTrue(self.unlocked(b["img2img"](1.0)))
        self.assertEqual(b["txt2img"](1.0), ("forge", False))

    def test_hires_negative_box_keeps_forges_rule(self):
        b = self.build([cfg_unlock._before_ui], [cfg_unlock._on_component])
        cfg_unlock.set_enabled("txt2img", True)
        self.assertIs(b["hires"], forge_use_cfg)

    def _check_chain(self, b, nag):
        for spd_t in (False, True):
            for spd_i in (False, True):
                for nag_t in (False, True):
                    for nag_i in (False, True):
                        cfg_unlock.set_enabled("txt2img", spd_t)
                        cfg_unlock.set_enabled("img2img", spd_i)
                        nag.enabled.update(txt2img=nag_t, img2img=nag_i)
                        flags = (spd_t, spd_i, nag_t, nag_i)
                        self.assertEqual(self.unlocked(b["txt2img"](1.0)), spd_t or nag_t, flags)
                        self.assertEqual(self.unlocked(b["img2img"](1.0)), spd_i or nag_i, flags)
                        # NAG's own variant still drives the hires box; ours does not
                        self.assertEqual(self.unlocked(b["hires"](1.0)), nag_t, flags)
                        for tab in ("txt2img", "img2img"):
                            self.assertTrue(self.unlocked(b[tab](4.0)), flags)

    def test_chains_with_nag_in_either_callback_order(self):
        for ours_first in (True, False):
            with self.subTest(ours_first=ours_first):
                self.setUp()
                nag = FakeNag(self.ui)
                before = [cfg_unlock._before_ui, nag.install]
                after = [cfg_unlock._on_component, nag.swap]
                if not ours_first:
                    before.reverse()
                    after.reverse()
                self._check_chain(self.build(before, after), nag)

    def test_survives_a_ui_reload(self):
        b = self.build([cfg_unlock._before_ui], [cfg_unlock._on_component])
        b = self.build([cfg_unlock._before_ui], [cfg_unlock._on_component])
        cfg_unlock.set_enabled("img2img", True)
        self.assertEqual(b["txt2img"](1.0), ("forge", False))
        self.assertTrue(self.unlocked(b["img2img"](1.0)))
        self.assertIs(b["hires"], forge_use_cfg)

    def test_reload_with_nag_ours_first_is_exact(self):
        nag = FakeNag(self.ui)
        before = [cfg_unlock._before_ui, nag.install]
        after = [cfg_unlock._on_component, nag.swap]
        self.build(before, after)
        self._check_chain(self.build(before, after), nag)

    def test_reload_with_nag_first_never_locks_wrongly(self):
        nag = FakeNag(self.ui)
        before = [nag.install, cfg_unlock._before_ui]
        after = [nag.swap, cfg_unlock._on_component]
        self.build(before, after)
        b = self.build(before, after)
        for spd_t in (False, True):
            for spd_i in (False, True):
                for nag_t in (False, True):
                    for nag_i in (False, True):
                        cfg_unlock.set_enabled("txt2img", spd_t)
                        cfg_unlock.set_enabled("img2img", spd_i)
                        nag.enabled.update(txt2img=nag_t, img2img=nag_i)
                        if spd_t or nag_t:
                            self.assertTrue(self.unlocked(b["txt2img"](1.0)))
                        if spd_i or nag_i:
                            self.assertTrue(self.unlocked(b["img2img"](1.0)))
                        if not any((spd_t, spd_i, nag_t, nag_i)):
                            # locked, by Forge's function or by the re-entry fallback
                            for tab in ("txt2img", "img2img"):
                                self.assertIn(b[tab](1.0), (("forge", False),
                                              ("update", {"interactive": False})))
                        for tab in ("txt2img", "img2img"):
                            self.assertTrue(self.unlocked(b[tab](4.0)) or
                                            b[tab](4.0) == ("update", {"interactive": True}))

    def test_unlock_update_needs_both_checkboxes(self):
        self.build([cfg_unlock._before_ui], [cfg_unlock._on_component])
        self.assertEqual(cfg_unlock.unlock_update("txt2img", True, True, 1.0),
                         ("update", {"interactive": True}))
        self.assertTrue(cfg_unlock.is_enabled("txt2img"))
        self.assertEqual(cfg_unlock.unlock_update("txt2img", True, False, 1.0), ("forge", False))
        self.assertFalse(cfg_unlock.is_enabled("txt2img"))
        self.assertEqual(cfg_unlock.unlock_update("txt2img", False, True, 4.0), ("forge", True))
        self.assertEqual(cfg_unlock.unlock_update("txt2img", False, True), "skip")
        self.assertFalse(cfg_unlock.is_enabled("img2img"))


class TestUnlockWiring(ForgeScriptTestCase):
    def setUp(self):
        super().setUp()
        accordion = sys.modules["modules.ui_components"].InputAccordion
        self.saved_side_effect = accordion.side_effect
        accordion.side_effect = lambda *a, **kw: MagicMock()
        cfg_unlock._enabled.update(txt2img=False, img2img=False)

    def tearDown(self):
        sys.modules["modules.ui_components"].InputAccordion.side_effect = self.saved_side_effect
        cfg_unlock._enabled.update(txt2img=False, img2img=False)

    def _build(self, is_img2img, with_cfg=True):
        script = speed_forge.SpeedForgeScript()
        prefix = "img2img" if is_img2img else "txt2img"
        neg = SimpleNamespace(elem_id=f"{prefix}_neg_prompt")
        cfg = SimpleNamespace(elem_id=f"{prefix}_cfg_scale")
        script.after_component(neg)
        script.after_component(SimpleNamespace(elem_id=f"{prefix}_distilled_cfg_scale"))
        if with_cfg:
            script.after_component(cfg)
        script.after_component(SimpleNamespace(elem_id=f"{prefix}_image_cfg_scale"))
        n_started = len(_registered["app_started"])
        components = script.ui(is_img2img)
        return script, components, neg, cfg, _registered["app_started"][n_started:]

    def test_both_checkboxes_push_the_update(self):
        script, components, neg, cfg, started = self._build(False)
        self.assertEqual(len(components), 18)
        enabled, diversity = components[0], components[13]
        self.assertIsNot(enabled, diversity)
        for checkbox in (enabled, diversity):
            checkbox.change.assert_called_once()
            kw = checkbox.change.call_args.kwargs
            self.assertEqual(kw["inputs"], [enabled, diversity, cfg])
            self.assertEqual(kw["outputs"], [neg])
            self.assertIs(kw["fn"].func, cfg_unlock.unlock_update)
            self.assertEqual(kw["fn"].args, ("txt2img",))

        # seeded from the saved values once the app has started
        self.assertEqual(len(started), 1)
        enabled.value, diversity.value = True, False
        started[0](None, None)
        self.assertFalse(cfg_unlock.is_enabled("txt2img"))
        diversity.value = True
        started[0](None, None)
        self.assertTrue(cfg_unlock.is_enabled("txt2img"))
        self.assertFalse(cfg_unlock.is_enabled("img2img"))

    def test_img2img_tab_and_missing_cfg_slider(self):
        script, components, neg, cfg, _ = self._build(True, with_cfg=False)
        kw = components[0].change.call_args.kwargs
        self.assertEqual(kw["inputs"], [components[0], components[13]])
        self.assertEqual(kw["fn"].args, ("img2img",))

    def test_run_resyncs_the_flag(self):
        script, _, _, _, _ = self._build(True)
        p = make_div_p(RecordingSampler())
        run_before(script, p, **DIV_UI)
        self.assertTrue(cfg_unlock.is_enabled("img2img"))
        run_before(script, p, **{**DIV_UI, "diversity_enabled": False})
        self.assertFalse(cfg_unlock.is_enabled("img2img"))
        run_before(script, p, enabled=False, **DIV_UI)
        self.assertFalse(cfg_unlock.is_enabled("img2img"))
        self.assertFalse(cfg_unlock.is_enabled("txt2img"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
