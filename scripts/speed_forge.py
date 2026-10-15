"""SPEED (Spectral Progressive Diffusion) for Forge Neo (sd-webui-forge-classic).

Based on https://github.com/howardhx/speed
Wraps the sampler selected in the main UI with the resolution transitions from
the paper: the sigma schedule is segmented at each transition and the latent is
spectrally expanded and timestep-aligned in between, so early steps run at a
fraction of the full resolution.

Integration: when enabled, ``process_before_every_sampling`` replaces
``p.sampler.func`` with a wrapper that delegates each segment to the original
sampler function (the math is shared with the ComfyUI node via
``speed_core.py`` at the extension root).

Designed for flow-matching models (Flux, Wan, ...) whose sigmas live in
(0, 1]; eps-prediction models (SD1.x/SDXL) run but are unvalidated.

The "RAW composition stage" subsection (Forge Neo only, see
docs/krea2-hybrid-plan.md) passes a ``segment_hook`` into the shared core:
coarse segments get the encoded negative prompt, a real CFG scale and the
step density of a RAW reference schedule; fine segments get Forge's own
``extra_args`` untouched.
"""
from __future__ import annotations

import functools
import importlib.util
import os
import sys
import types

import gradio as gr
import torch

from modules import errors, scripts
from modules.ui_components import InputAccordion

EXT_ROOT = scripts.basedir()

_PKG_NAME = "comfyui_speed_forge_pkg"


def _log(msg: str):
    # ASCII-only messages; guard anyway for legacy Windows console codepages.
    try:
        print(f"[SPEED] {msg}")
    except UnicodeEncodeError:
        print(f"[SPEED] {msg.encode('ascii', 'replace').decode('ascii')}")


def _load_speed_core():
    """Load ``speed_core`` (and its ``spectral_utils`` dependency) from the
    extension root under a private synthetic package, so the relative imports
    used by the ComfyUI side resolve without polluting generic module names."""
    full = f"{_PKG_NAME}.speed_core"
    if full in sys.modules:
        return sys.modules[full]
    pkg = types.ModuleType(_PKG_NAME)
    pkg.__path__ = [EXT_ROOT]
    sys.modules[_PKG_NAME] = pkg
    try:
        for name in ("spectral_utils", "speed_core"):
            mod_name = f"{_PKG_NAME}.{name}"
            spec = importlib.util.spec_from_file_location(
                mod_name, os.path.join(EXT_ROOT, f"{name}.py"))
            module = importlib.util.module_from_spec(spec)
            sys.modules[mod_name] = module
            spec.loader.exec_module(module)
    except Exception:
        for name in ("spectral_utils", "speed_core"):
            sys.modules.pop(f"{_PKG_NAME}.{name}", None)
        sys.modules.pop(_PKG_NAME, None)
        raise
    return sys.modules[full]


speed_core = _load_speed_core()


class _ShapeSafeRandn:
    """Wrapper for ``k_diffusion.sampling.torch`` (Forge's TorchHijack).

    TorchHijack.randn_like ignores the input shape and returns the next
    pre-generated full-resolution noise tensor, which breaks ancestral/SDE
    samplers while a SPEED segment runs at a coarse resolution. Fall back to
    seeded CPU noise whenever the hijacked shape does not match.
    """

    def __init__(self, inner, seed: int):
        self._inner = inner
        self._generator = torch.Generator(device="cpu").manual_seed(int(seed) % (2**31 - 1))

    def __getattr__(self, item):
        if item == "randn_like":
            return self.randn_like
        return getattr(self._inner, item)

    def randn_like(self, x):
        try:
            noise = self._inner.randn_like(x)
        except Exception:
            noise = None
        if isinstance(noise, torch.Tensor) and noise.shape == x.shape:
            return noise
        return torch.randn(x.shape, generator=self._generator,
                           dtype=torch.float32).to(device=x.device, dtype=x.dtype)


def _anima_reference_latents_active(p) -> bool:
    """True when Anima will concatenate reference latents onto ``x`` mid-sampling.

    Anima's ``forward`` (``backend/nn/anima.py``) does
    ``torch.cat((x, ref), dim=2)`` -- the *temporal* axis -- so every reference
    has to match ``x`` spatially. A SPEED coarse segment runs ``x`` at a
    fraction of the full latent size while the stored references stay at full
    size, and the cat raises. That covers Anima Edit and img2img referencing,
    both gated behind the ``anima_do_reference`` setting.

    Detected from the runtime list rather than the setting alone, so a run that
    happens to carry no references still gets SPEED. ``dynamic_args.ref_latents``
    is rewritten on every positive-prompt conditioning pass (cleared when the
    setting is off), which happens before ``process_before_every_sampling``, so
    it cannot go stale between runs.

    Scoped to Anima on purpose: Flux/Krea/Qwen populate the same list but
    consume it differently, and those paths are not what this guard was written
    against.
    """
    model = getattr(p, "sd_model", None)
    if model is None or type(model).__name__ != "Anima":
        return False
    try:
        from backend.args import dynamic_args
    except ImportError:
        return False
    return bool(getattr(dynamic_args, "ref_latents", None))


# =============================================================================
# RAW composition stage ("diversity" subsection)
# =============================================================================

# Model class name -> experimental. Krea 2 is the tested target; Z-Image and
# FLUX.2 Klein have community Turbo LoRAs and run uncalibrated.
DIVERSITY_MODELS = {"Krea2": False, "ZImage": True, "Flux2": True}

# Tolerance for "this sigma is on the reference schedule". Schedules are float32.
_SIGMA_TOL = 1e-4


def _parse_patterns(text) -> tuple:
    """Comma-separated, case-insensitive filename substrings."""
    return tuple(t.strip().lower() for t in str(text or "").split(",") if t.strip())


# --- Turbo LoRA: per-file online routing -------------------------------------
#
# Forge bakes prompt LoRAs into the weights at extra_networks.activate through
# networks.load_lora_for_models. Loaded with online_mode=True instead, each
# patch becomes an OnlineLoRAPatch in the patcher's weight_wrapper_patches,
# re-applied on every module forward from its mutable payload, so writing
# ``obj.patch[0][0]`` changes the next model call. That is what Forge's own
# LoRA Control Integrated does per step; this routes only the files matching
# the Turbo pattern (the way Neo-LoraCtl intercepts the loader) and drives
# their strength per segment. Every other LoRA reaches the stock loader with
# its original arguments.

_original_load_lora_for_models = None

# [strength, adapter, strength_model, offset, function] (backend/patcher/base.py)
_PATCH_PAYLOAD_LEN = 5


def _lora_matches(filename, patterns) -> bool:
    base = os.path.splitext(os.path.basename(str(filename)))[0].lower()
    return any(pat in base for pat in patterns)


def _lora_label(filename) -> str:
    return os.path.splitext(os.path.basename(str(filename)))[0]


def _null_lora_hash(p):
    """Make the stock loader rebuild prompt LoRAs on the next activation."""
    models = [getattr(p, "sd_model", None)]
    try:
        from modules import shared
        models.append(getattr(shared, "sd_model", None))
    except ImportError:
        pass
    for model in models:
        if model is not None and hasattr(model, "current_lora_hash"):
            model.current_lora_hash = None


def _install_lora_interception() -> bool:
    global _original_load_lora_for_models
    if _original_load_lora_for_models is not None:
        return True
    try:
        import networks
    except ImportError:
        _log("diversity: Forge's builtin LoRA module ('networks') is not importable; "
             "the RAW composition stage cannot switch the Turbo LoRA.")
        return False
    _original_load_lora_for_models = networks.load_lora_for_models
    networks.load_lora_for_models = _intercepted_load_lora_for_models
    return True


def _validate_patch_object(obj) -> str | None:
    """Error string if ``obj`` does not look like the OnlineLoRAPatch this was
    written against (same check as Neo-LoraCtl's), else None."""
    patch = getattr(obj, "patch", None)
    if not isinstance(patch, list) or len(patch) < 1:
        return f"patch object has no 1+-element .patch list: {type(obj).__name__}"
    entry = patch[0]
    if not isinstance(entry, (tuple, list)) or len(entry) != _PATCH_PAYLOAD_LEN:
        got = len(entry) if isinstance(entry, (tuple, list)) else type(entry).__name__
        return f"patch payload arity mismatch: expected {_PATCH_PAYLOAD_LEN}, got {got}"
    if not isinstance(entry[0], (int, float)) or isinstance(entry[0], bool):
        return f"patch strength is not a number: {type(entry[0]).__name__}"
    if not callable(obj):
        return f"patch object is not callable: {type(obj).__name__}"
    return None


def _set_patch_strength(obj, strength: float):
    entry = obj.patch[0]
    if isinstance(entry, list):
        entry[0] = strength
    else:
        obj.patch[0] = (strength,) + tuple(entry[1:])


def _set_turbo_strength(entries, scale: float):
    """Write ``base * scale`` into every adopted Turbo payload."""
    for obj, base, _label in entries:
        _set_patch_strength(obj, base * scale)


def _global_online_lora() -> bool:
    try:
        from backend.args import dynamic_args
    except ImportError:
        return False
    return bool(getattr(dynamic_args, "online_lora", False))


def _turbo_owner(filename, online_mode) -> str | None:
    """Why another writer already owns this file's patch strengths, or None.

    Two writers on ``patch[0][0]`` would fight every step, so the feature
    stands down rather than share: Forge's builtin LoRA Control (its
    ``<lora:name:[a:b]>`` mapping, keyed by filename), Neo-LoraCtl when its
    filter schedules this file (its default filter excludes accelerator
    LoRAs), or any outer interceptor that already asked for online loading
    while the global on-the-fly option is off.
    """
    for data in getattr(scripts, "scripts_data", None) or []:
        script_class = getattr(data, "script_class", None)
        name = getattr(script_class, "__name__", "")
        if name == "LoRAControl" and filename in (getattr(script_class, "mapping", None) or {}):
            return "Forge's LoRA Control Integrated schedules it (<lora:...:[a:b]> syntax)"
        if name == "NeoLoraCtlScript" and getattr(script_class, "enabled", False):
            loractl_core = sys.modules.get("neo_loractl_core")
            check = getattr(loractl_core, "lora_is_scheduled", None)
            try:
                scheduled = check is not None and check(
                    filename, list(getattr(script_class, "patterns", []) or []),
                    getattr(script_class, "filter_mode", "exclude"))
            except Exception:
                scheduled = False
            if scheduled:
                return "Neo-LoraCtl schedules it (its filter includes this file)"
    if online_mode and not _global_online_lora():
        return "another extension already loads it on the fly"
    return None


def _intercepted_load_lora_for_models(model, clip, lora, strength_model, strength_clip,
                                      filename="default", online_mode=False):
    cls = SpeedForgeScript
    if cls.turbo_route is not None and not cls.turbo_cycle_open:
        # First load call of a fresh activation: the stock loader has just
        # reset forge_objects to the originals, so prior adoptions are stale.
        cls.turbo_cycle_open = True
        cls.turbo_entries = []
        cls.turbo_blockers = []
    if (cls.turbo_route is None or model is None
            or not _lora_matches(filename, cls.turbo_route)):
        return _original_load_lora_for_models(model, clip, lora, strength_model,
                                              strength_clip, filename, online_mode)

    label = _lora_label(filename)
    owner = _turbo_owner(filename, online_mode)
    if owner is not None:
        cls.turbo_blockers.append(f"'{label}': {owner}")
        return _original_load_lora_for_models(model, clip, lora, strength_model,
                                              strength_clip, filename, online_mode)

    wrappers_before = getattr(model, "weight_wrapper_patches", None)
    before = {k: len(v) for k, v in (wrappers_before or {}).items()}
    result = _original_load_lora_for_models(model, clip, lora, strength_model,
                                            strength_clip, filename, True)
    if result is None:
        return result
    new_model = result[0]
    wrappers = getattr(new_model, "weight_wrapper_patches", None)
    if wrappers_before is None or wrappers is None:
        cls.turbo_blockers.append(f"'{label}': the model has no weight_wrapper_patches "
                                  "(Forge API drift?)")
        return result

    adopted, errors = [], []
    for key, objs in wrappers.items():
        for obj in objs[before.get(key, 0):]:
            err = _validate_patch_object(obj)
            if err is not None:
                errors.append(f"{key}: {err}")
            else:
                adopted.append((obj, float(obj.patch[0][0]), label))
    if errors:
        # The LoRA stays online at its prompt strength: the same output as a
        # plain <lora:> tag, applied on the fly. The feature stands down.
        cls.turbo_blockers.append(f"'{label}': refusing to drive its patches (Forge API "
                                  f"drift?): {errors[0]}"
                                  + (f" (+{len(errors) - 1} more)" if len(errors) > 1 else ""))
        return result
    if not adopted:
        _log(f"diversity: Turbo LoRA '{label}' added no patches to this model.")
        return result
    cls.turbo_entries.extend(adopted)
    return result


def _redensify(seg_sigmas, reference, anchor):
    """Replace one coarse segment's sigmas with the reference schedule's.

    ``reference`` is the RAW reference schedule (floats, descending).
    ``anchor`` is ``None`` for the first coarse segment, whose start must then
    be on the reference schedule (a txt2img run starts at sigma_max on both);
    for a later coarse segment it is the previous segment's unpatched end,
    because this segment starts at SPEED's aligned ``t_tilde``, which is on no
    schedule. The segment keeps its own first and last sigma; in between go
    the reference entries strictly inside its bounds.

    Returns ``(sigmas, None)`` or ``(None, reason)``.
    """
    first, last = float(seg_sigmas[0]), float(seg_sigmas[-1])
    if anchor is None:
        if not any(abs(r - first) <= _SIGMA_TOL for r in reference):
            return None, (f"segment starts at sigma {first:.4f}, which is not on the "
                          f"{len(reference) - 1}-step reference schedule (img2img or a "
                          "mid-schedule start); keeping this segment's own steps")
        anchor = first
    upper = min(first, anchor)
    values = ([first]
              + [r for r in reference if last + _SIGMA_TOL < r < upper - _SIGMA_TOL]
              + [last])
    if len(values) < 2 or any(b >= a for a, b in zip(values, values[1:])):
        return None, ("reference slice is not strictly decreasing; keeping this "
                      "segment's own steps")
    return torch.tensor(values, dtype=seg_sigmas.dtype, device=seg_sigmas.device), None


def _make_index_map(orig_len: int, new_len: int):
    """Map a callback index of the re-densified segment onto the original
    segment's range, forwarding only the first call per original index, so the
    progress bar and Forge's step counters see one callback per original step."""
    last = [-1]

    def index_map(j):
        i = min(j * orig_len // new_len, orig_len - 1)
        if i <= last[0]:
            return None
        last[0] = i
        return i

    return index_map


def _encode_negative(p):
    """The unconditional at CFG 1, through the same call ``setup_conds``
    (modules/processing.py) makes in its CFG > 1 branch, and so through Forge's
    own ``cached_uc``: a batch encodes once, and the result is the scheduled
    list ``reconstruct_cond_batch`` indexes by step."""
    from modules import prompt_parser
    negative_prompts = prompt_parser.SdConditioning(
        p.negative_prompts, width=p.width, height=p.height, is_negative_prompt=True,
        distilled_cfg_scale=p.distilled_cfg_scale)
    return p.get_conds_with_caching(
        prompt_parser.get_learned_conditioning, negative_prompts, p.firstpass_steps,
        [p.cached_uc], p.extra_network_data)


def _guidance_overrides_note(p) -> str | None:
    """Denoiser options that can force the scale back to 1.0 after the wrapper
    set it (sd_samplers_cfg_denoiser.py): Skip Early CFG and NGMS."""
    try:
        from modules.shared import opts
        skip_early = float(getattr(opts, "skip_early_cond", 0.0) or 0.0)
    except Exception:
        skip_early = 0.0
    ngms = float(getattr(p, "s_min_uncond", 0.0) or 0.0)
    notes = []
    if skip_early > 0.0:
        notes.append(f"Skip Early CFG = {skip_early:g}")
    if ngms > 0.0:
        notes.append(f"NGMS = {ngms:g}")
    if not notes:
        return None
    return (" and ".join(notes) + " is set; it can drop the RAW CFG back to 1.0 "
            "on some coarse steps")


def _make_diversity_hook(div: dict, model, schedule_end: float, run: dict):
    """Build the ``segment_hook`` for one sampling pass.

    ``div`` is the per-pass runtime built in ``process_before_every_sampling``
    (``uc``, ``cfg``, ``reference``); ``run`` collects what happened, for the
    restore and the summary in ``_finish_diversity``.
    """
    prev_end = [None]

    def hook(seg_index, scale, seg_sigmas, extra_args):
        coarse = scale < 1.0 - 1e-6
        if coarse and abs(float(seg_sigmas[-1]) - schedule_end) <= 1e-9:
            # No transition follows: SPEED is running the rest of the schedule
            # at reduced resolution. The full-resolution stage this feature
            # hands off to never comes, so it stands down for the segment.
            if not run.get("no_transition_logged"):
                run["no_transition_logged"] = True
                _log("diversity: no full-resolution segment follows; RAW stage not "
                     "applied to this pass")
            _set_turbo_strength(div["entries"], 1.0)
            return seg_sigmas, extra_args, None
        if not coarse:
            # Turbo LoRA back at the prompt strength before the first fine call.
            _set_turbo_strength(div["entries"], 1.0)
            run["fine_segments"] = run.get("fine_segments", 0) + 1
            return seg_sigmas, extra_args, None

        anchor = prev_end[0]
        prev_end[0] = float(seg_sigmas[-1])
        orig_len = len(seg_sigmas) - 1
        new_sigmas, reason = _redensify(seg_sigmas, div["reference"], anchor)
        index_map = None
        if new_sigmas is None:
            _log(f"diversity: {reason}")
            new_sigmas = seg_sigmas
        else:
            new_len = len(new_sigmas) - 1
            if new_len != orig_len:
                index_map = _make_index_map(orig_len, new_len)
            if new_len < orig_len:
                _log(f"diversity: the {len(div['reference']) - 1}-step reference is "
                     f"sparser than this run here ({new_len} vs {orig_len} steps)")
            _bump_total_steps(model, new_len - orig_len, run)
        new_len = len(new_sigmas) - 1

        seg_args = dict(extra_args)
        seg_args["uncond"] = div["uc"]
        seg_args["cond_scale"] = div["cfg"]
        _set_turbo_strength(div["entries"], 0.0)

        run["coarse_segments"] = run.get("coarse_segments", 0) + 1
        run["coarse_steps"] = run.get("coarse_steps", 0) + new_len
        _log(f"diversity: coarse segment {seg_index} at scale {scale:g}: {orig_len} -> "
             f"{new_len} steps, sigma {float(new_sigmas[0]):.4f}->"
             f"{float(new_sigmas[-1]):.4f}, CFG {div['cfg']:g}")
        return new_sigmas, seg_args, index_map

    return hook


def _bump_total_steps(model, extra_steps: int, run: dict):
    """``CFGDenoiser.total_steps`` is compared with its per-call ``step`` for
    Skip Early CFG, NGMS parity and the refiner switch; add the extra calls a
    re-densified segment makes. Restored by ``_finish_diversity``."""
    total = getattr(model, "total_steps", None)
    if not isinstance(total, int) or extra_steps == 0:
        return
    if "total_steps_backup" not in run:
        run["total_steps_backup"] = total
    steps = getattr(model, "steps", None)
    mult = max(1, round(total / steps)) if isinstance(steps, int) and steps > 0 else 1
    model.total_steps = max(1, total + extra_steps * mult)


def _finish_diversity(div: dict, model, run: dict):
    """Undo the pass-scoped changes. Runs on success and on interruption, so
    an interrupted coarse segment does not leave the Turbo LoRA at 0."""
    _set_turbo_strength(div["entries"], 1.0)
    if "total_steps_backup" in run and model is not None:
        model.total_steps = run["total_steps_backup"]


def _diversity_infotext(div: dict, run: dict) -> str:
    parts = [f"ref={div['ref_steps']}", f"cfg={div['cfg']:g}"]
    if div.get("lora_label"):
        parts.append(f"lora={div['lora_label']}")
    parts.append(f"coarse={run.get('coarse_steps', 0)}")
    return (f"diversity({', '.join(parts)})"
            + (" experimental" if div.get("experimental") else ""))


def _make_speed_func(base_fn, cfg: dict):
    """Wrap a k-diffusion ``sample_*`` function with SPEED segmentation."""

    @functools.wraps(base_fn)
    def sample_speed_wrapped(model, x, sigmas=None, extra_args=None,
                             callback=None, disable=None, **kwargs):
        if not isinstance(sigmas, torch.Tensor):
            # dpm_fast-style solvers receive n/sigma_min/sigma_max instead of
            # a schedule; SPEED cannot segment those, run the original as-is.
            return base_fn(model, x, sigmas, extra_args=extra_args,
                           callback=callback, disable=disable, **kwargs)

        if float(sigmas[0]) > 1.5:
            _log(f"note: sigma_max={float(sigmas[0]):.2f} > 1; SPEED's timestep "
                 "alignment assumes flow-matching sigmas in (0, 1] (Flux/Wan). "
                 "Results on eps-prediction models are unvalidated.")

        # Objects tied to the full-resolution latent shape must not reach
        # coarse segments (Brownian noise sampler pre-built from full-res x).
        full_res_kwargs = {}
        noise_sampler = kwargs.pop("noise_sampler", None)
        if noise_sampler is not None:
            full_res_kwargs["noise_sampler"] = noise_sampler

        _log(f"pass start: x={tuple(x.shape)}, steps={len(sigmas) - 1}, "
             f"sigmas {float(sigmas[0]):.4f}->{float(sigmas[-2]):.4f}, "
             f"mode={cfg['mode_label']}, transform={cfg['transform']}, "
             f"scales={cfg['scales']}, seed={cfg['seed']}")

        div = cfg.get("diversity")
        hook = None
        run = {}
        if div is not None:
            hook = _make_diversity_hook(div, model, float(sigmas[-1]), run)

        import k_diffusion.sampling as kds
        prev_torch = kds.torch
        kds.torch = _ShapeSafeRandn(prev_torch, cfg["seed"])
        try:
            result = speed_core.sample_speed_core(
                base_fn, model, x, sigmas,
                extra_args=extra_args, callback=callback, disable=disable,
                transform=cfg["transform"], mode=cfg["mode"],
                scales=cfg["scales"], delta=cfg["delta"],
                spectrum_A=cfg["spectrum_A"], spectrum_beta=cfg["spectrum_beta"],
                manual_sigmas=cfg["manual_sigmas"], seed=cfg["seed"],
                adaptive_delta=cfg["adaptive_delta"], ref_latent=cfg["ref_latent"],
                ref_shift=cfg["ref_shift"], sigma_divisor=cfg["sigma_divisor"],
                sampler_kwargs=kwargs, full_res_sampler_kwargs=full_res_kwargs,
                log_fn=_log, segment_hook=hook,
            )
        finally:
            kds.torch = prev_torch
            if div is not None:
                _finish_diversity(div, model, run)

        if div is not None:
            if run.get("coarse_segments"):
                p = div["p"]
                p.extra_generation_params["SPEED"] = (
                    f"{div['infotext_base']} {_diversity_infotext(div, run)}")
            elif not run.get("no_transition_logged"):
                _log("diversity: no coarse segment ran in this pass (it starts below "
                     "the transition sigma); RAW stage not applied")

        SpeedForgeScript.completed_passes += 1
        return result

    sample_speed_wrapped._speed_wrapped = True
    return sample_speed_wrapped


class SpeedForgeScript(scripts.Script):
    # Per-run state as CLASS attributes (persists across instances/tabs; reset
    # both when disabled and in postprocess).
    enabled: bool = False
    config: dict | None = None
    apply_hr: bool = False
    wrapped_passes: int = 0
    completed_passes: int = 0
    # RAW composition stage: settings for this run when it is active (feature
    # on, SPEED on, supported model), else None.
    diversity: dict | None = None
    # Turbo LoRA routing. ``turbo_route`` is the pattern tuple the loader
    # interception routes online this run (None = pass everything through);
    # ``turbo_membership`` is the route of the last run, to null the stock
    # loader's hash only when it changes. Adopted entries are (OnlineLoRAPatch,
    # base strength) and deliberately persist across runs: the stock loader
    # reuses its LoRA application while its hash is unchanged, and so do the
    # live objects. Staleness is checked against the live unet per pass.
    turbo_route: tuple | None = None
    turbo_membership: tuple | None = None
    turbo_cycle_open: bool = False
    turbo_entries: list = []      # (OnlineLoRAPatch, base strength, LoRA name)
    turbo_blockers: list = []

    def title(self):
        return "SPEED (Spectral Progressive Diffusion)"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with InputAccordion(False, label=self.title()) as enabled:
            gr.Markdown(
                "Runs early denoising steps at a reduced latent resolution, then "
                "spectrally expands to full resolution. The sampler selected above "
                "is used as the base solver. Designed for flow-matching models "
                "(Flux, Wan, Krea 2, Z-Image). Not applied to inpainting."
            )
            with gr.Row():
                transform = gr.Radio(
                    choices=["dct", "dwt", "fft"], value="dct", label="Transform",
                    info="Spectral basis at each transition. DWT needs 2x scale steps and PyWavelets.",
                )
                mode = gr.Radio(
                    choices=["neo_shift", "delta_optimal", "manual"], value="neo_shift",
                    label="Mode",
                    info="neo_shift: delta_optimal plus the Sigma divisor below, tuned for "
                         "this webui's fixed (non-resolution-shifted) Flux/FLUX.2 schedule; "
                         "delta_optimal: power-spectrum preset as calibrated upstream; "
                         "manual: explicit sigma thresholds.",
                )
                model_preset = gr.Dropdown(
                    choices=list(speed_core._PRESETS.keys()), value="flux",
                    label="Spectrum preset",
                    info="Power-spectrum fit used by delta_optimal mode. krea-2 = few-step "
                         "Krea 2 (Turbo, or base + Turbo LoRA); krea-2-raw = base model at "
                         "40-50 steps with CFG; z-image = Z-Image Turbo; flux2 = FLUX.2 at "
                         "40-50 steps; anima = Anima at its 32-step preset. flux/flux2 are "
                         "calibrated from measured output; the rest are spectrum-derived "
                         "and untested.",
                )
                # Persisted normally (no do_not_save_to_config): a preset this
                # extension no longer offers doesn't need special-casing here.
                # Forge Neo's own ui_loadsave.py already guards every gr.Dropdown
                # against restoring a stale value -- `check_dropdown` in
                # modules/ui_loadsave.py only restores a saved value that is
                # still in the component's current `choices`; otherwise it
                # silently keeps this constructor's `value="flux"` default. And
                # even in the hypothetical it didn't, `before_process_batch`'s
                # `_PRESETS.get(model_preset)` already treats an unrecognized
                # name as "custom" rather than erroring.
            with gr.Row():
                scales = gr.Textbox(
                    value="0.5,1.0", label="Scales",
                    info="Comma-separated resolution fractions ending at 1.0, e.g. 0.25,0.5,1.0",
                )
                with gr.Column():
                    adaptive_delta = gr.Checkbox(
                        value=True, label="Adaptive delta",
                        info="Hold the coarse/detail step split at the preset's 1024x1024 "
                             "calibration instead of letting the coarse pass grow with "
                             "resolution. No-op at or below 1024x1024.",
                    )
                    delta = gr.Slider(
                        minimum=0.0001, maximum=0.5, step=0.0001, value=0.01, label="Delta",
                        info="Noise-dominated tolerance; smaller transitions later "
                             "(delta_optimal). With Adaptive delta on, this is the value "
                             "at the 1024x1024 reference.",
                    )
                    sigma_divisor = gr.Slider(
                        minimum=0.5, maximum=2.0, step=0.01, value=1.03,
                        label="Sigma divisor",
                        info="neo_shift mode only. Divides the resolved sigma* by this "
                             "value before the step lookup. 1.0 exactly matches "
                             "delta_optimal (mathematically a no-op). Higher = lower "
                             "threshold = more coarse steps = more speed (and more "
                             "quality risk); lower = fewer coarse steps = more "
                             "conservative. Default 1.03 was measured on this webui's "
                             "flat (non-resolution-shifted) Flux schedule at 1024x1024, "
                             "Euler/Beta -- re-check it if your sampler/scheduler differ.",
                    )
            with gr.Row():
                manual_sigmas = gr.Textbox(
                    value="0.85", label="Manual sigmas",
                    info="One threshold per transition, strictly decreasing (manual mode).",
                )
                seed = gr.Number(
                    value=-1, precision=0, label="Spectral noise seed",
                    info="-1 = follow the generation seed.",
                )
            with gr.Row():
                spectrum_A = gr.Number(
                    value=203.615097, label="Spectrum A",
                    info="Power-law amplitude (custom preset only).",
                )
                spectrum_beta = gr.Number(
                    value=1.915461, label="Spectrum beta",
                    info="Power-law decay exponent (custom preset only).",
                )
            apply_hr = gr.Checkbox(
                value=False, label="Apply to hires-fix pass (experimental)",
                visible=not is_img2img,
            )
            with InputAccordion(False, label="Diversity: RAW composition stage (Krea 2)") \
                    as diversity_enabled:
                gr.Markdown(
                    "For a RAW checkpoint with a Turbo LoRA in the prompt. While SPEED "
                    "runs its reduced-resolution steps, the Turbo LoRA is held at "
                    "strength 0, the negative prompt is applied with real CFG, and "
                    "those steps run at the density the RAW model expects. From the "
                    "first full-resolution step on, the Turbo LoRA returns at its "
                    "prompt strength and sampling continues at the main CFG. Stands "
                    "down when no prompt LoRA matches the pattern. Tested on Krea 2; "
                    "Z-Image and FLUX.2 Klein run as experimental. Step-scheduled "
                    "prompt syntax shifts its switch points while this is on."
                )
                with gr.Row():
                    diversity_ref_steps = gr.Slider(
                        minimum=28, maximum=52, step=1, value=28, label="Reference steps",
                        info="Steps the RAW model would run on its own (28 = diffusers "
                             "default, 52 = official repo). Only the part above SPEED's "
                             "transition sigma runs.",
                    )
                    diversity_cfg = gr.Slider(
                        minimum=1.0, maximum=10.0, step=0.5, value=4.5, label="RAW CFG",
                        info="CFG for the reduced-resolution steps, standard convention "
                             "(4.5 here equals Krea's 3.5).",
                    )
                    diversity_pattern = gr.Textbox(
                        value="turbo", label="Turbo LoRA pattern",
                        info="Comma-separated, case-insensitive filename substrings. "
                             "Only matching prompt LoRAs are switched; all others keep "
                             "their prompt strength.",
                    )

        return [enabled, transform, mode, model_preset, scales, adaptive_delta,
                delta, sigma_divisor, manual_sigmas, spectrum_A, spectrum_beta,
                seed, apply_hr, diversity_enabled, diversity_ref_steps,
                diversity_cfg, diversity_pattern]

    @staticmethod
    def _resolve_diversity(p, enabled, ref_steps, raw_cfg, pattern):
        """Settings dict for this run, or None (feature off or standing down)."""
        if not enabled:
            return None
        model_class = type(getattr(p, "sd_model", None)).__name__
        if model_class not in DIVERSITY_MODELS:
            _log(f"diversity: RAW composition stage is for Krea 2 (and, experimental, "
                 f"Z-Image / FLUX.2 Klein); this model is {model_class}, standing down.")
            return None
        patterns = _parse_patterns(pattern)
        if not patterns:
            _log("diversity: Turbo LoRA pattern is empty; standing down.")
            return None
        experimental = DIVERSITY_MODELS[model_class]
        if experimental:
            _log(f"diversity: RAW composition stage on {model_class} is experimental "
                 "(uncalibrated; tested on Krea 2 only).")
        return {
            "ref_steps": int(ref_steps),
            "cfg": float(raw_cfg),
            "patterns": patterns,
            "experimental": experimental,
            "model_class": model_class,
        }

    def before_process_batch(self, p, *args, **kwargs):
        # Runs before extra_networks.activate (modules/processing.py), which is
        # where the Turbo LoRA gets loaded, so the route set here is the one
        # the loader interception sees for this batch.
        self._configure_run(p, *args, **kwargs)
        self._update_turbo_route(p)

    @staticmethod
    def _update_turbo_route(p):
        cls = SpeedForgeScript
        cls.turbo_cycle_open = False
        route = cls.diversity["patterns"] if cls.diversity is not None else None
        if route is not None and not _install_lora_interception():
            cls.diversity = None
            route = None
        cls.turbo_route = route
        if route != cls.turbo_membership:
            # The route decides how the Turbo LoRA is loaded (online or baked)
            # but is not part of the stock loader's hash, so a change has to
            # force the reload. Unchanged, the loader's cache stays valid.
            cls.turbo_membership = route
            cls.turbo_entries = []
            cls.turbo_blockers = []
            _null_lora_hash(p)

    @staticmethod
    def _live_turbo_entries(p):
        """Adopted entries whose objects are in the unet about to sample."""
        unet = getattr(getattr(getattr(p, "sd_model", None), "forge_objects", None),
                       "unet", None)
        wrappers = getattr(unet, "weight_wrapper_patches", None)
        if not wrappers:
            return []
        live = {id(obj) for objs in wrappers.values() for obj in objs}
        return [e for e in SpeedForgeScript.turbo_entries if id(e[0]) in live]

    def _configure_run(self, p, enabled, transform, mode, model_preset,
                       scales, adaptive_delta, delta, sigma_divisor,
                       manual_sigmas, spectrum_A, spectrum_beta, seed,
                       apply_hr, diversity_enabled=False, diversity_ref_steps=28,
                       diversity_cfg=4.5, diversity_pattern="turbo",
                       *args, **kwargs):
        cls = SpeedForgeScript
        cls.enabled = bool(enabled)
        cls.config = None
        cls.apply_hr = bool(apply_hr)
        cls.wrapped_passes = 0
        cls.completed_passes = 0
        cls.diversity = None
        if not cls.enabled:
            return

        preset = speed_core._PRESETS.get(model_preset)
        if preset is not None:
            A, beta = preset["A"], preset["beta"]
            ref_latent = preset["ref_latent"]
            ref_shift = preset.get("ref_shift")   # absent once a preset is measured-calibrated
        else:
            A, beta = float(spectrum_A), float(spectrum_beta)
            # Unknown model: assume the 8x VAE case, and leave ref_shift unset
            # so adaptive falls back to its flat backstop.
            ref_latent = speed_core.REFERENCE_LATENT
            ref_shift = None

        # neo_shift is delta_optimal plus sigma_divisor; it is a Forge Neo UI
        # label only, never passed to speed_core -- see speed_core.py's
        # `_PRESETS` comment block for why the correction lives here instead
        # of in the shared preset table.
        core_mode = "delta_optimal" if mode == "neo_shift" else mode
        resolved_divisor = float(sigma_divisor) if mode == "neo_shift" else 1.0

        try:
            parsed_scales = speed_core._parse_scales(scales)
            parsed_sigmas = (speed_core._parse_sigmas(manual_sigmas, upper=None)
                             if mode == "manual" else [])
        except ValueError as e:
            cls.enabled = False
            errors.report(f"SPEED: invalid settings, disabled for this run: {e}")
            return

        seed = int(seed)
        if seed < 0:
            batch_seeds = kwargs.get("seeds") or [0]
            seed = int(batch_seeds[0])

        cls.config = {
            "transform": transform,
            "mode": core_mode,
            "mode_label": mode,   # raw UI selection, for logging/metadata only
            "scales": parsed_scales,
            "delta": float(delta),
            "spectrum_A": A,
            "spectrum_beta": beta,
            "manual_sigmas": parsed_sigmas,
            "seed": seed,
            "adaptive_delta": bool(adaptive_delta),
            "ref_latent": int(ref_latent),
            "ref_shift": ref_shift,
            "sigma_divisor": resolved_divisor,
        }

        p.extra_generation_params["SPEED"] = (
            f"{transform}/{mode}/{model_preset} scales={scales} delta={delta} "
            + ("adaptive " if adaptive_delta and core_mode == "delta_optimal" else "")
            + (f"sigma_divisor={sigma_divisor} " if mode == "neo_shift" else "")
            + (f"manual_sigmas={manual_sigmas} " if mode == "manual" else "")
            + f"seed={seed}"
        )

        cls.diversity = self._resolve_diversity(
            p, diversity_enabled, diversity_ref_steps, diversity_cfg, diversity_pattern)

    @staticmethod
    def _diversity_runtime(p, div: dict, uc):
        """Per-pass state for the segment hook, or None to stand down this pass.

        Built here rather than inside the sampler wrapper because both calls
        must happen outside sampling: the encode goes through Forge's cond
        cache, and ``get_sigmas`` is wrapped by other extensions (Neo-LoraCtl
        captures the schedule from it), so calling it during the pass would
        overwrite what they captured for the real schedule.
        """
        cls = SpeedForgeScript
        if cls.turbo_blockers:
            _log("diversity: Turbo LoRA " + "; ".join(cls.turbo_blockers)
                 + ". RAW composition stage skipped for this run.")
            return None
        entries = cls._live_turbo_entries(p)
        cls.turbo_entries = entries
        if not entries:
            _log(f"diversity: no prompt LoRA matches the Turbo pattern "
                 f"'{', '.join(div['patterns'])}'; RAW composition stage skipped for "
                 "this run (it needs a Turbo LoRA to switch off).")
            return None

        if uc is None:
            try:
                uc = _encode_negative(p)
            except Exception as e:
                _log(f"diversity: could not encode the negative prompt ({e}); RAW stage "
                     "skipped for this pass.")
                return None
        try:
            reference = [float(s) for s in p.sampler.get_sigmas(p, div["ref_steps"])]
        except Exception as e:
            _log(f"diversity: could not build the {div['ref_steps']}-step reference "
                 f"schedule ({e}); RAW stage skipped for this pass.")
            return None
        note = _guidance_overrides_note(p)
        if note:
            _log(f"diversity: {note}.")
        return {
            **div,
            "uc": uc,
            "reference": reference,
            "entries": entries,
            "lora_label": "+".join(f"{name}:{base:g}" for name, base in
                                   dict.fromkeys((e[2], e[1]) for e in entries)),
            "p": p,
            "infotext_base": p.extra_generation_params.get("SPEED", ""),
        }

    def process_before_every_sampling(self, p, *args, **kwargs):
        cls = SpeedForgeScript
        if not cls.enabled or cls.config is None:
            return

        if getattr(p, "is_hr_pass", False) and not cls.apply_hr:
            return

        if getattr(p, "mask", None) is not None or getattr(p.sd_model, "is_inpaint", False):
            _log("inpainting detected; SPEED skipped (masks/image conditioning "
                 "are tied to the full latent resolution).")
            return

        if _anima_reference_latents_active(p):
            _log("Anima reference latents detected (Anima Edit / img2img "
                 "referencing); SPEED skipped -- the references are concatenated "
                 "onto the latent and are tied to its full resolution. Turn off "
                 "'anima_do_reference' in Settings > Stable Diffusion to use "
                 "SPEED on this model.")
            return

        sampler = getattr(p, "sampler", None)
        if sampler is None or not hasattr(sampler, "func") or not callable(sampler.func):
            _log("no compatible sampler function found on this pass; SPEED skipped.")
            return
        if getattr(sampler.func, "_speed_wrapped", False):
            return

        pass_cfg = dict(cls.config)
        if cls.diversity is not None and not getattr(p, "is_hr_pass", False):
            pass_cfg["diversity"] = self._diversity_runtime(p, cls.diversity, kwargs.get("uc"))

        sampler.func = _make_speed_func(sampler.func, pass_cfg)
        cls.wrapped_passes += 1

    def postprocess(self, p, processed, *args):
        cls = SpeedForgeScript
        if cls.enabled:
            if cls.wrapped_passes == 0:
                _log("enabled but no sampling pass was wrapped; nothing was applied.")
            elif cls.completed_passes == 0:
                _log("sampler was wrapped but the SPEED function never ran; "
                     "check the console for earlier errors.")
        # Park the Turbo LoRA at its prompt strength between generations. The
        # wrapper already does this per pass; this covers a run that ended
        # some other way. Entries persist (see the class attributes).
        _set_turbo_strength(cls.turbo_entries, 1.0)
        cls.turbo_cycle_open = False
        cls.enabled = False
        cls.config = None
        cls.apply_hr = False
        cls.wrapped_passes = 0
        cls.completed_passes = 0
        cls.diversity = None
