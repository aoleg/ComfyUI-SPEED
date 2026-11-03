"""ComfyUI sampler for Spectral Progressive Diffusion.

Based on https://github.com/howardhx/speed
Thin ComfyUI adapter over ``speed_core.sample_speed_core``: resolves the base
solver the way KSamplerSelect does (``comfy.samplers.sampler_object``), or
takes any SAMPLER object, and exposes the node schema. The framework-agnostic
math lives in ``speed_core.py`` (shared with the Forge Neo webui script in
``scripts/speed_forge.py``).
"""
from __future__ import annotations

from typing import List

import torch

import comfy.samplers
from comfy_api.latest import io
import comfy.k_diffusion.sampling as kds

from .speed_core import (
    _PRESETS,
    REFERENCE_LATENT,
    _parse_scales,
    _parse_sigmas,
    sample_speed_core,
)


def _log(msg: str):
    print(f"[SPEED] {msg}")


def _init_image_part(model, x, sigmas):
    """Return the init image's contribution to ``x``, or ``None`` for txt2img.

    ComfyUI's ``KSAMPLER.sample`` hands the sampler function a
    ``KSamplerX0Inpaint`` whose ``latent_image`` is the clean latent (already
    through ``process_latent_in``, zeros for an empty latent) and whose
    ``inner_model`` is the guider holding the ``BaseModel``. The model's
    ``noise_scaling`` with zero noise gives exactly the image part of ``x`` in
    the model's own convention (``(1 - sigma) * latent`` for flow models).
    """
    latent = getattr(model, "latent_image", None)
    if not isinstance(latent, torch.Tensor) or tuple(latent.shape) != tuple(x.shape):
        return None
    if not torch.count_nonzero(latent):
        return None
    latent = latent.to(device=x.device, dtype=x.dtype)
    try:
        model_sampling = model.inner_model.inner_model.model_sampling
        return model_sampling.noise_scaling(sigmas[0], torch.zeros_like(x), latent, False)
    except Exception:
        s0 = float(sigmas[0])
        return latent * (1.0 - s0) if s0 <= 1.0 else latent


def _list_samplers() -> List[str]:
    """Sampler names: the list KSamplerSelect shows, which other packs extend.

    Read when the schema is built, so a pack loaded before this one (RES4LYF
    inserts its names into this list) is included. Falls back to the
    ``sample_*`` functions of the k-diffusion module.
    """
    try:
        names = list(comfy.samplers.KSampler.SAMPLERS)
    except Exception:
        names = []
    if not names:
        names = [a[len("sample_"):] for a in dir(kds) if a.startswith("sample_")]
    return names


def _resolve_base(base_sampler: str, sampler=None):
    """Return ``(sampler_object, label)`` for the base solver.

    ``sampler`` is a connected SAMPLER object and wins over the name. A name
    goes through ``comfy.samplers.sampler_object``, which also covers the
    samplers that are not plain ``sample_*`` functions (``uni_pc``, ``ddim``,
    ``dpm_fast``) and whatever other packs registered.
    """
    if sampler is None:
        try:
            sampler = comfy.samplers.sampler_object(base_sampler)
        except Exception as e:
            raise ValueError(f"Unknown base sampler {base_sampler!r}: {e}") from e
        label = base_sampler
    else:
        fn = getattr(sampler, "sampler_function", None)
        label = getattr(fn, "__name__", type(sampler).__name__).removeprefix("sample_")
    fn = getattr(sampler, "sampler_function", None)
    if not callable(fn):
        raise ValueError(
            f"SAMPLER {type(sampler).__name__} has no k-diffusion style sampler_function; "
            "SPEED needs one to segment the schedule."
        )
    own_sample = getattr(type(sampler), "sample", None)
    if own_sample is not None and own_sample is not comfy.samplers.KSAMPLER.sample:
        _log(f"SAMPLER {type(sampler).__name__} overrides KSAMPLER.sample; SPEED runs its "
             "sampler_function per segment and that override does not apply.")
    return sampler, label


@torch.no_grad()
def sample_speed(
    model, x, sigmas, extra_args=None, callback=None, disable=None,
    *,
    base=None,
    base_label: str = "euler",
    transform: str = "dct",
    mode: str = "delta_optimal",
    scales: List[float] = None,
    delta: float = 0.01,
    spectrum_A: float = 203.615097,
    spectrum_beta: float = 1.915461,
    manual_sigmas: List[float] = None,
    seed: int = 0,
    adaptive_delta: bool = True,
    ref_latent: int = 128,
    ref_shift: float = None,
    log: bool = True,
    noise_fill: str = "seed",
):
    """Comfy-compatible ``sample_*`` function.

    ``base`` is the resolved base SAMPLER object: its ``sampler_function``
    runs each segment with its ``extra_options`` as keyword arguments.
    """
    if base is None:
        base, base_label = _resolve_base(base_label)
    sampler_fn = base.sampler_function
    sampler_kwargs = dict(getattr(base, "extra_options", {}) or {})

    if extra_args is not None and extra_args.get("denoise_mask") is not None:
        # A noise mask is blended against the full-resolution latent after
        # every step; a coarse segment cannot honour it. Run the base sampler.
        if log:
            _log("noise mask present (masked / inpaint denoising); SPEED skipped for this pass.")
        return sampler_fn(model, x, sigmas, extra_args=extra_args,
                          callback=callback, disable=disable, **sampler_kwargs)

    init_image = _init_image_part(model, x, sigmas)
    if log:
        _log(f"pass start: x={tuple(x.shape)}, steps={len(sigmas) - 1}, "
             f"sigmas {float(sigmas[0]):.4f}->{float(sigmas[-2]):.4f}, "
             f"sampler={base_label}, mode={mode}, transform={transform}, "
             f"scales={scales}, seed={seed}"
             + (", img2img" if init_image is not None else ""))

    return sample_speed_core(
        sampler_fn, model, x, sigmas,
        extra_args=extra_args, callback=callback, disable=disable,
        transform=transform, mode=mode, scales=scales, delta=delta,
        spectrum_A=spectrum_A, spectrum_beta=spectrum_beta,
        manual_sigmas=manual_sigmas, seed=seed,
        adaptive_delta=adaptive_delta, ref_latent=ref_latent,
        ref_shift=ref_shift,
        sampler_kwargs=sampler_kwargs,
        log_fn=_log if log else None,
        init_image=init_image,
        noise_fill=noise_fill,
    )


# =============================================================================
# Comfy node
# =============================================================================

class SamplerSPEED(io.ComfyNode):
    """Spectral Progressive Diffusion sampler node."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SamplerSPEED",
            display_name="Sampler SPEED (Spectral Progressive Diffusion)",
            category="sampling/custom_sampling/samplers",
            inputs=[
                io.Combo.Input(
                    "base_sampler", options=_list_samplers(), default="euler",
                    tooltip="Underlying solver, from the same list KSamplerSelect offers "
                            "(other packs' samplers included). Multistep solvers restart "
                            "at each SPEED transition because the schedule is segmented. "
                            "Ignored when a SAMPLER is connected to `sampler`.",
                ),
                io.Combo.Input(
                    "transform", options=["dct", "dwt", "fft"], default="dct",
                    tooltip="Spectral basis used at each transition. DCT (default) "
                            "supports any scale ratio; DWT requires consecutive scales "
                            "to differ by exactly 2x; FFT accepts any ratio.",
                ),
                io.Combo.Input(
                    "mode", options=["delta_optimal", "manual"], default="delta_optimal",
                    tooltip="`delta_optimal` computes transitions from `scales`, "
                            "`delta`, and the selected power-spectrum preset. "
                            "`manual` uses user-specified sigma thresholds.",
                ),
                io.Combo.Input(
                    "model_preset", options=list(_PRESETS.keys()), default="flux",
                    tooltip="Power-spectrum preset for delta-optimal mode. `flux` and "
                            "`flux2` are calibrated from measured A/B output (sigma* 0.968); "
                            "`wan21` is the upstream published fit. `krea-2` targets "
                            "few-step Krea 2 (the Turbo checkpoint at ~8 steps, or the "
                            "base model with the Turbo LoRA at ~20 steps); `krea-2-raw` "
                            "targets the undistilled base model at 40-50 steps with CFG "
                            "and transitions later for more speedup; `z-image` targets "
                            "Z-Image Turbo; `flux2` targets FLUX.2 [dev]/Klein at 40-50 "
                            "steps (its beta is borrowed from Flux.1 -- no measured "
                            "spectrum fit exists for FLUX.2's VAE). `custom` uses the "
                            "manual A / beta inputs below. sigma* is quoted at 1024x1024; "
                            "see `log` for where a run actually landed.",
                ),
                io.String.Input(
                    "scales", default="0.5,1.0",
                    tooltip="Comma-separated resolution fractions ending at 1.0. Used in "
                            "delta_optimal mode. Example: `0.5,1.0` or `0.25,0.5,1.0`.",
                ),
                io.Boolean.Input(
                    "adaptive_delta", default=True,
                    tooltip="Keep the coarse/detail step split at the split the preset "
                            "was calibrated for (1024x1024) instead of letting it grow "
                            "with the canvas. Eq. 10 evaluates the spectrum at the "
                            "coarse grid's own Nyquist frequency, so without this the "
                            "same `delta` spends progressively more of the schedule at "
                            "low resolution as you generate larger -- for `flux` at 40 "
                            "steps, 25/40 coarse at 1536x1536 becomes 29/40 unadapted, "
                            "which starves the detail pass exactly where you need it. "
                            "Only ever transitions earlier than the unadapted path, "
                            "never later, so it is a no-op at or below 1024x1024. "
                            "Applies to delta_optimal mode only.",
                ),
                io.Float.Input(
                    "delta", default=0.01, min=1e-4, max=0.5, step=0.001,
                    tooltip="Noise-dominated tolerance. Smaller values transition later. "
                            "With `adaptive_delta` on this is the value at the 1024x1024 "
                            "reference -- exactly the delta the preset table is "
                            "documented against, applied at any resolution.",
                ),
                io.String.Input(
                    "manual_sigmas", default="0.85",
                    tooltip="Comma-separated sigma thresholds, one per transition (length "
                            "S-1 to match `scales`). Used in `manual` mode. Example "
                            "for scales=`0.25,0.5,1.0`: `0.95,0.85`.",
                ),
                io.Float.Input(
                    "spectrum_A", default=203.615097, min=0.0, max=1e6, step=0.001,
                    tooltip="Power-spectrum amplitude A (used when model_preset=custom).",
                ),
                io.Float.Input(
                    "spectrum_beta", default=1.915461, min=0.0, max=10.0, step=0.001,
                    tooltip="Power-spectrum decay exponent beta (used when model_preset=custom).",
                ),
                io.Int.Input(
                    "seed", default=0, min=0, max=2**31 - 1, step=1,
                    tooltip="Seed for the spectral-noise padding at each transition, when "
                            "`noise_fill` is `fresh` (and always for the dwt transform).",
                ),
                io.Boolean.Input(
                    "log", default=True,
                    tooltip="Print the resolved transition step(s) and sigma(s) to the "
                            "console every run, e.g. `[SPEED] transitions: step 8: "
                            "0.5->1 (sigma=0.8473)`. Cheap (one line per generation) and "
                            "the fastest way to tell whether a preset/delta is landing "
                            "the transition where you expect at your actual resolution "
                            "and step count.",
                ),
                io.Combo.Input(
                    "noise_fill", options=["seed", "fresh"], default="seed",
                    tooltip="Where the noise that fills each newly opened band at a "
                            "transition comes from. `seed`: the run's own seed noise, so "
                            "every band of every stage carries the noise a plain run at "
                            "this seed uses and the output matches that run as closely as "
                            "the coarse pass allows. `fresh`: new noise from `seed` at "
                            "each transition (the dwt transform always does this).",
                ),
                io.Sampler.Input(
                    "sampler", optional=True,
                    tooltip="Optional base solver as a SAMPLER object (KSamplerSelect, "
                            "RES4LYF's ClownSampler, any custom sampler node). When "
                            "connected it replaces `base_sampler`: its sampler function "
                            "runs each segment with its own options.",
                ),
            ],
            outputs=[io.Sampler.Output()],
        )

    @classmethod
    def execute(cls, base_sampler, transform, mode, model_preset, scales,
                adaptive_delta, delta, manual_sigmas, spectrum_A, spectrum_beta,
                seed, log, noise_fill="seed", sampler=None):
        preset = _PRESETS.get(model_preset)
        if preset is not None:
            A, beta = preset["A"], preset["beta"]
            ref_latent = preset["ref_latent"]
            ref_shift = preset.get("ref_shift")   # absent once a preset is measured-calibrated
        else:
            A, beta = float(spectrum_A), float(spectrum_beta)
            # Unknown model: assume the 8x VAE case, and leave ref_shift unset
            # so adaptive falls back to its flat backstop.
            ref_latent = REFERENCE_LATENT
            ref_shift = None

        parsed_scales = _parse_scales(scales)
        parsed_sigmas = _parse_sigmas(manual_sigmas) if mode == "manual" else []
        base, base_label = _resolve_base(base_sampler, sampler)

        wrapped = comfy.samplers.KSAMPLER(
            sample_speed,
            extra_options={
                "base": base,
                "base_label": base_label,
                "transform": transform,
                "mode": mode,
                "scales": parsed_scales,
                "delta": float(delta),
                "spectrum_A": A,
                "spectrum_beta": beta,
                "manual_sigmas": parsed_sigmas,
                "seed": int(seed),
                "adaptive_delta": bool(adaptive_delta),
                "ref_latent": int(ref_latent),
                "ref_shift": ref_shift,
                "log": bool(log),
                "noise_fill": noise_fill,
            },
            # ddim is euler with ComfyUI's random inpaint noise; keep whatever
            # the base object asked for.
            inpaint_options=dict(getattr(base, "inpaint_options", {}) or {}),
        )
        return io.NodeOutput(wrapped)
