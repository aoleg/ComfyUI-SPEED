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
             f"mode={cfg['mode']}, transform={cfg['transform']}, "
             f"scales={cfg['scales']}, seed={cfg['seed']}")

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
                ref_shift=cfg["ref_shift"],
                sampler_kwargs=kwargs, full_res_sampler_kwargs=full_res_kwargs,
                log_fn=_log,
            )
        finally:
            kds.torch = prev_torch

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
                    choices=["delta_optimal", "manual"], value="delta_optimal", label="Mode",
                    info="delta_optimal: from power-spectrum preset; manual: explicit sigma thresholds.",
                )
                model_preset = gr.Dropdown(
                    choices=list(speed_core._PRESETS.keys()), value="flux",
                    label="Spectrum preset",
                    info="Power-spectrum fit used by delta_optimal mode. krea-2 = few-step "
                         "Krea 2 (Turbo, or base + Turbo LoRA); krea-2-raw = base model at "
                         "40-50 steps with CFG; z-image = Z-Image Turbo; flux2 = FLUX.2 at "
                         "40-50 steps. flux/flux2 are calibrated from measured output; "
                         "the rest are spectrum-derived and untested.",
                )
                # The preset list is defined by this extension, so keep a stale
                # ui-config.json entry from restoring a value we no longer offer.
                model_preset.do_not_save_to_config = True
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

        return [enabled, transform, mode, model_preset, scales, adaptive_delta,
                delta, manual_sigmas, spectrum_A, spectrum_beta, seed, apply_hr]

    def before_process_batch(self, p, enabled, transform, mode, model_preset,
                             scales, adaptive_delta, delta, manual_sigmas,
                             spectrum_A, spectrum_beta, seed, apply_hr,
                             *args, **kwargs):
        cls = SpeedForgeScript
        cls.enabled = bool(enabled)
        cls.config = None
        cls.apply_hr = bool(apply_hr)
        cls.wrapped_passes = 0
        cls.completed_passes = 0
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
            "mode": mode,
            "scales": parsed_scales,
            "delta": float(delta),
            "spectrum_A": A,
            "spectrum_beta": beta,
            "manual_sigmas": parsed_sigmas,
            "seed": seed,
            "adaptive_delta": bool(adaptive_delta),
            "ref_latent": int(ref_latent),
            "ref_shift": ref_shift,
        }

        p.extra_generation_params["SPEED"] = (
            f"{transform}/{mode}/{model_preset} scales={scales} delta={delta} "
            + ("adaptive " if adaptive_delta and mode == "delta_optimal" else "")
            + (f"manual_sigmas={manual_sigmas} " if mode == "manual" else "")
            + f"seed={seed}"
        )

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

        sampler = getattr(p, "sampler", None)
        if sampler is None or not hasattr(sampler, "func") or not callable(sampler.func):
            _log("no compatible sampler function found on this pass; SPEED skipped.")
            return
        if getattr(sampler.func, "_speed_wrapped", False):
            return

        sampler.func = _make_speed_func(sampler.func, dict(cls.config))
        cls.wrapped_passes += 1

    def postprocess(self, p, processed, *args):
        cls = SpeedForgeScript
        if cls.enabled:
            if cls.wrapped_passes == 0:
                _log("enabled but no sampling pass was wrapped; nothing was applied.")
            elif cls.completed_passes == 0:
                _log("sampler was wrapped but the SPEED function never ran; "
                     "check the console for earlier errors.")
        cls.enabled = False
        cls.config = None
        cls.apply_hr = False
        cls.wrapped_passes = 0
        cls.completed_passes = 0
