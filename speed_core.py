"""Framework-agnostic core for SPEED (Spectral Progressive Diffusion).

Based on https://github.com/howardhx/speed
Shared by the ComfyUI node (``speed_sampler.py``) and the Forge Neo webui
script (``scripts/speed_forge.py``). This module must not import ComfyUI or
webui modules — only numpy/torch and ``spectral_utils``.

``sample_speed_core`` wraps any k-diffusion style ``sample_*`` solver with the
resolution transitions from the paper. The denoising trajectory is segmented
at each transition; between segments the latent is spectrally expanded and
timestep-aligned.
"""
from __future__ import annotations

import math
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch

from .spectral_utils import (
    _dct_expand_np,
    _dwt_expand_np,
    _fft_expand_np,
    align_timestep,
    delta_optimal_transitions,
    equivalent_delta,
    kappa,
    reference_coarse_fraction,
    validate_scales,
)


# =============================================================================
# Per-model power-spectrum presets.
#
# ``A`` and ``beta`` describe the latent power spectrum ``P(w) = A * w**-beta``;
# ``delta_optimal`` mode turns ``P`` at the coarse grid's Nyquist frequency into
# the transition time via Eq. 9. ``flux`` and ``wan21`` are the fits shipped by
# the official repo.
#
# The remaining presets are *calibrated*, not measured: ``beta`` is taken from
# the measured fit for the same VAE latent space, and ``A`` is then solved so
# that the default ``delta=0.01`` with ``scales=0.5,1.0`` at 1024x1024 places
# the transition at a target sigma chosen from the model's step budget. Each
# entry below records that target; ``tests/test_math.py`` re-derives it.
#
#   preset       latent space          sigma* @1024   basis
#   flux         Flux VAE              0.9680         measured (see below)
#   wan21        Wan 2.1 VAE           0.6941         upstream published fit
#   krea-2       Wan 2.1 / Qwen-Image  0.9000         spectrum-derived, untested
#   krea-2-raw   Wan 2.1 / Qwen-Image  0.8300         spectrum-derived, untested
#   z-image      Flux VAE              0.9400         spectrum-derived, untested
#   flux2        Flux VAE (borrowed)   0.9680         measured (ref latent 64)
#
# ``flux`` and ``flux2`` are calibrated from A/B output, and **``flux`` no
# longer ships the published fit** (A 203.615097, beta 1.915461, sigma* 0.8527
# -- restorable via the ``custom`` preset). The measurement that moved it: on
# Flux.1 dev at 1024x1024, transitioning at sigma 0.9389 renders license-plate
# text as wrong letters, while 0.9611 renders it correctly. The published
# threshold of 0.8527 is far below both.
#
# What made this findable is that the quality boundary is a *sigma*, and it
# holds across models, resolutions and schedules that differ by 7.6x:
#
#   FLUX.2 1536px  15/40 coarse  sigma 0.9915  indistinguishable from off
#   Flux.1 1536px  15/40 coarse  sigma 0.9915  text fine
#   Flux.1 1024px  12/40 coarse  sigma 0.9611  text correct
#   Flux.1 1024px  15/40 coarse  sigma 0.9389  text broken
#   FLUX.2 1536px  29/40 coarse  sigma 0.9362  sharp, small details missing
#   FLUX.2 1536px  33/40 coarse  sigma 0.8628  coherent but visibly simpler
#   FLUX.2 1536px  35/40 coarse  sigma 0.7762  broken
#
# Everything at sigma >= 0.96 passes. Note the step counts do *not* transfer --
# Flux.1 needed 28 full-resolution steps at 1024px but only 25 at 1536px,
# because fine detail lands on more pixels as the canvas grows. Only the sigma
# is invariant, which is why the threshold rather than the step budget is what
# these presets pin. 0.968 is the anchor: it reproduces both measured optima
# (11-12 of 40 coarse at 1024px, 25 of 40 at 1536px).
#
# A consequence worth knowing: under a fixed sigma threshold SPEED's payoff
# *grows* with resolution, because a larger canvas runs a more compressed
# schedule and so puts more steps above the threshold. Measured with ``flux``
# at 40 steps: ~1.15x at 1024px on a stock schedule, ~1.9x at 1536px.
#
# The other four are untouched pending their own A/B and keep both the
# spectrum-derived threshold and the step-fraction cap that compensates for it.
#
# Why the Krea-2 presets are not just ``wan21``: Krea 2 encodes with the
# Qwen-Image VAE, whose latent format (16ch, 8x, same per-channel mean/std) is
# the Wan 2.1 one, so ``beta`` carries over — but the ``wan21`` amplitude was
# fitted on low-resolution video latents and puts the transition at sigma 0.69,
# which on Krea 2's schedule spends 5 of 8 Turbo steps at half resolution. A
# 1024px still holds far more high-frequency latent energy than a 480p video
# frame, so a larger ``A`` (earlier activation) is the physically expected
# direction as well as the empirically better one.
#
# Krea 2 also runs a strongly shifted schedule (mu=1.15, i.e. an effective
# shift of exp(1.15)=3.158 at every resolution; some ComfyUI workflows raise it
# to 1.5 -> 4.48), which holds sigma near 1.0 for the first third of the run.
# A threshold that reads as "early" therefore lands mid-schedule: at 8 steps,
# sigma* 0.90 leaves 5 of 8 steps at full resolution, and 0.8527 (the ``flux``
# preset) leaves 4-5 depending on the scheduler.
#
# Z-Image shares the Flux VAE latent space, so ``beta`` is Flux's; only the
# amplitude is re-calibrated, because Z-Image Turbo runs 8-9 steps on a much
# stronger shift (3.0 in ComfyUI, 9.0 in the Forge Neo preset) and the ``flux``
# threshold leaves as few as 2 full-resolution steps at shift 9. sigma* 0.94
# keeps at least 5 of 9 steps at full resolution across that whole shift range.
#
# ``flux2`` is the weakest-founded entry here and is flagged as such: there is
# no measured (A, beta) fit for FLUX.2's VAE anywhere this repo has access to,
# and unlike Krea 2 it is not a byte-identical match to an existing fit --
# FLUX.2's latent is 32 real channels at 16x spatial compression (packed into
# a 128-channel/8x tensor; see ``latent_rgb_factors_reshape`` in ComfyUI's
# ``latent_formats.py``), not the 16ch/8x of Flux.1's. ``beta`` is Flux.1's,
# on no stronger basis than "closest available analog, same lineage" -- take
# it as a starting point to refine with ``custom``, not a calibrated value.
# What *is* source-backed: FLUX.2 Klein 4B/9B both declare ``shift = 2.02``
# (ComfyUI ``supported_models.py``, log-space mu like Flux.1/Krea 2, i.e. an
# effective multiplier of exp(2.02) = 7.54 -- much stronger than Flux.1's
# default 1.15). No separate entry exists for the 32B "dev" checkpoint at
# this writing; Klein's value is assumed to carry over since all FLUX.2
# variants share the same VAE. That shift crowds sigma near 1.0 hard: at 40-50
# steps even sigma* 0.90 -- deliberately the least aggressive target in this
# table -- lands the transition close to the schedule's midpoint.
#
# ``delta`` remains the speed/quality slider on top of a preset. For ``krea-2``
# it spans sigma* 0.93 (delta=0.005) to 0.80 (delta=0.05); smaller delta means
# an earlier transition, higher quality and less speedup.
#
# ``ref_latent`` is the latent short side the preset's sigma* is calibrated at,
# i.e. what a 1024x1024 generation looks like on that model's VAE grid. It is
# 128 for the 8x-compression VAEs (Flux, Wan 2.1 / Qwen-Image, Z-Image) and 64
# for FLUX.2, whose sampler-visible latent is 128 channels at 1/16 scale: its
# VAE encodes at 8x into 32 channels and then packs 2x2 spatially into the
# channel axis (``AutoencoderKLFlux2.encode``, ``patch_size: [2, 2]``), so a
# 1024px image is a 64x64 grid, not 128x128. ``adaptive_delta`` uses this to
# hold the coarse/detail step split steady across resolutions -- see
# ``_resolve_transitions``.
# =============================================================================

# Latent short side of the reference case every preset's sigma* is quoted at
# (1024x1024 on an 8x-compression VAE). Also the ``adaptive_delta`` fallback
# for ``custom``, where the model -- and so its compression -- is unknown.
REFERENCE_LATENT = 128

# Degeneration backstop, not a budget target: the coarse segments may not
# consume more than this fraction of the schedule. It exists only so an
# extreme schedule cannot starve the full-resolution pass to nothing -- at 40
# steps it guarantees 8 full-resolution steps, and measured output only breaks
# below about 6. It is deliberately loose enough never to fire on a
# configuration anyone runs; a *calibrated* sigma threshold is what actually
# decides the split. Applies to presets with no ``ref_shift`` (see below).
MAX_COARSE_FRACTION = 0.8

# ``ref_shift`` is the model's own effective schedule multiplier at the
# reference resolution, and its units differ by predictor family: it is
# ``exp(mu)`` for the ``PredictionFlux`` models (Flux's mu 1.15 -> 3.158,
# FLUX.2's 2.02 -> 7.538) and the raw multiplier for the
# ``PredictionDiscreteFlow`` ones (Z-Image 3.0, Wan 2.1 8.0). Mixing the two
# up is the trap documented in knowledge_sigmas.md 5.4. It converts the
# reference transition sigma into a reference *step fraction*, which
# ``_resolve_transitions`` then holds.
#
# **A preset that carries no ``ref_shift`` has been calibrated against measured
# output and does not want that step-fraction cap** -- its sigma is already the
# right answer, and capping the share would override it (for ``flux`` the cap
# would demand 5 of 40 steps where the measured-correct answer is 11). ``flux``
# and ``flux2`` are in that state; the rest still run the spectrum-derived
# threshold and keep the cap that compensates for it.
_PRESETS = {
    "flux":       {"A": 1963.48906, "beta": 1.915461, "ref_latent": 128},
    "wan21":      {"A": 219.484718, "beta": 2.422687, "ref_latent": 128, "ref_shift": 8.0},
    "krea-2":     {"A": 2357.960557, "beta": 2.422687, "ref_latent": 128, "ref_shift": 3.158193},
    "krea-2-raw": {"A": 887.402358, "beta": 2.422687, "ref_latent": 128, "ref_shift": 3.158193},
    "z-image":    {"A": 877.001476, "beta": 1.915461, "ref_latent": 128, "ref_shift": 3.0},
    "flux2":      {"A": 520.495848, "beta": 1.915461, "ref_latent": 64},
    "custom": None,
}


# =============================================================================
# Parsing helpers
# =============================================================================

def _parse_scales(s: str) -> List[float]:
    """Parse a comma-separated scale list."""
    out = [float(x.strip()) for x in s.split(",") if x.strip()]
    validate_scales(out)
    return out


def _parse_sigmas(s: str, upper: Optional[float] = 1.0) -> List[float]:
    """Parse comma-separated manual transition sigmas.

    ``upper`` bounds each sigma exclusively; pass ``None`` to allow arbitrary
    positive thresholds (eps-prediction schedules go well above 1.0).
    """
    out = [float(x.strip()) for x in s.split(",") if x.strip()]
    if upper is not None:
        if any(not (0.0 < v < upper) for v in out):
            raise ValueError(f"every manual sigma must be in (0, {upper}); got {out}")
    elif any(v <= 0.0 for v in out):
        raise ValueError(f"every manual sigma must be > 0; got {out}")
    for a, b in zip(out[:-1], out[1:]):
        if not (a > b):
            raise ValueError(f"manual sigmas must be strictly decreasing; got {out}")
    return out


# =============================================================================
# Spectral transition helper.
# =============================================================================

def _expand_and_align_torch(
    x: torch.Tensor, s_i: float, s_next: float, t: float,
    transform: str, seed: int, H_full: int, W_full: int,
) -> Tuple[torch.Tensor, float]:
    """Expand a 4D image latent or 5D video latent over its spatial axes."""
    if transform not in ("dct", "dwt", "fft"):
        raise ValueError(f"transform must be dct|dwt|fft, got {transform!r}")
    r = s_next / s_i
    H_tgt = round(s_next * H_full)
    W_tgt = round(s_next * W_full)

    if x.ndim == 5:
        B, C, T_frames, h_lo, w_lo = x.shape
        x4 = x.permute(0, 2, 1, 3, 4).reshape(B * T_frames, C, h_lo, w_lo)
    elif x.ndim == 4:
        x4 = x
    else:
        raise ValueError(f"expected 4D or 5D latent, got shape {tuple(x.shape)}")

    x_np = x4.detach().cpu().float().numpy()
    if transform == "dwt":
        if abs(r - 2.0) > 1e-6:
            raise ValueError(
                f"DWT requires r=2 between consecutive scales; got r={r:.4f}. "
                "Use transform=dct or transform=fft for non-dyadic ratios."
            )
        expanded = _dwt_expand_np(x_np, t, seed)
    elif transform == "dct":
        expanded = _dct_expand_np(x_np, (H_tgt, W_tgt), t, seed)
    else:
        expanded = _fft_expand_np(x_np, (H_tgt, W_tgt), t, seed)

    rescaled = (kappa(t, r) * expanded).astype(np.float32)
    x4_new = torch.from_numpy(rescaled).to(device=x.device, dtype=x.dtype)

    if x.ndim == 5:
        out = x4_new.reshape(B, T_frames, C, H_tgt, W_tgt).permute(0, 2, 1, 3, 4)
    else:
        out = x4_new

    return out, align_timestep(t, r)


# =============================================================================
# Initial coarse-resolution latent.
# =============================================================================

def _initial_dct_downscale(x: torch.Tensor, scale: float) -> torch.Tensor:
    """Downscale ``x`` by DCT truncation."""
    if scale >= 1.0:
        return x

    H_full, W_full = x.shape[-2], x.shape[-1]
    H_lo, W_lo = round(H_full * scale), round(W_full * scale)

    if x.ndim == 5:
        B, C, T_frames, _, _ = x.shape
        x4 = x.permute(0, 2, 1, 3, 4).reshape(B * T_frames, C, H_full, W_full)
    else:
        x4 = x

    x_np = x4.detach().cpu().float().numpy()
    from scipy.fft import dctn, idctn
    out_np = np.empty(x_np.shape[:-2] + (H_lo, W_lo), dtype=np.float32)
    for idx in np.ndindex(*x_np.shape[:-2]):
        coeffs = dctn(x_np[idx], type=2, norm="ortho")
        out_np[idx] = idctn(coeffs[:H_lo, :W_lo], type=2, norm="ortho").astype(np.float32)
    out4 = torch.from_numpy(out_np).to(device=x.device, dtype=x.dtype)

    if x.ndim == 5:
        return out4.reshape(B, T_frames, C, H_lo, W_lo).permute(0, 2, 1, 3, 4)
    return out4


# =============================================================================
# Transition scheduling.
# =============================================================================

def _resolve_transitions(
    sigmas: torch.Tensor, scales: List[float], delta: float, A: float, beta: float,
    H_full: int, W_full: int,
    adaptive: bool = False, ref_latent: int = REFERENCE_LATENT,
    ref_shift: float = None,
    log_fn: Callable[[str], None] = None,
) -> List[Tuple[int, float, float]]:
    """Return ``(step_idx, s_i, s_next)`` transitions from ``scales`` and ``delta``.

    With ``adaptive`` set, the transition timing is held at what it would be for
    a ``ref_latent``-sized latent instead of tracking the actual one.

    Why that is wanted: Eq. 10 evaluates the spectrum at the *coarse grid's own
    Nyquist frequency* in latent pixels, so the same preset and delta transition
    progressively later as the canvas grows -- for ``flux`` at 40 steps, 16 of
    40 steps coarse at 1024x1024 but 22 of 40 at 2048x2048. That is Eq. 10
    working correctly (a bigger canvas really does keep its high-frequency band
    noise-dominated for longer), but it inverts the point of the extension: the
    coarse pass only lays down composition, and giving it a *larger* share of
    the schedule at high resolution buys nothing while starving the detail
    pass. Adaptive removes the resolution term so the coarse/detail split stays
    where the preset was calibrated, with no per-resolution retuning.

    The threshold is pinned as ``t* = max(t*_reference, t*_actual)``. That
    removes the resolution term while staying one-directional: below the
    reference resolution the actual threshold is already the earlier
    (higher-sigma, more conservative) one and is kept, so adaptive is a no-op
    there rather than making small canvases more aggressive -- at 512x512 a 0.5
    first scale is a 256px coarse pass, marginal already.

    For a preset with a *measured* sigma (no ``ref_shift``) that is the whole
    mechanism, and it is enough: a sigma is invariant across schedules, so the
    step split it produces adapts on its own. Only ``MAX_COARSE_FRACTION``
    remains, as a backstop against degeneration rather than a budget.

    For a preset still on the spectrum-derived threshold (``ref_shift`` set),
    the sigma alone lands too late once the schedule is compressed -- ComfyUI's
    ``ModelSamplingFlux`` and SwarmUI's Sigma Shift derive mu from width*height,
    and a FLUX.2-class schedule at 1536x1536 holds sigma above 0.93 for 30 of 40
    steps. Those presets additionally hold the coarse *share* at
    ``ceil(n * reference_coarse_fraction(t*, ref_shift))``, the share the
    reference case spends coarse, which tracks the preset and ``delta``. Without
    it such a run gives 33/40 coarse, and at 3072x3072 the transition never
    fires at all and the output stays half-resolution.
    """
    if len(scales) < 2:
        return []
    t_stars = delta_optimal_transitions(scales, delta, A, beta, H_full, W_full)
    n_steps = len(sigmas) - 1

    if adaptive:
        t_ref = delta_optimal_transitions(scales, delta, A, beta, ref_latent, ref_latent)
        pinned = [max(actual, reference) for actual, reference in zip(t_stars, t_ref)]
        if log_fn is not None and any(
            abs(a - b) > 1e-9 for a, b in zip(t_stars, pinned)
        ):
            omega_max = min(H_full, W_full) / 2.0
            eq = equivalent_delta(pinned[0], A, beta, scales[0] * omega_max)
            log_fn(
                f"adaptive delta: latent {H_full}x{W_full} vs reference "
                f"{ref_latent}x{ref_latent}; sigma* "
                f"{', '.join(f'{a:.4f}->{b:.4f}' for a, b in zip(t_stars, pinned))} "
                f"(delta {delta:g} -> {eq:.6g} equivalent at this resolution)"
            )
        t_stars = pinned

    n_transitions = len(scales) - 1
    out: List[Tuple[int, float, float]] = []
    for i, (s_old, s_new, t_thr) in enumerate(zip(scales[:-1], scales[1:], t_stars)):
        step_idx = next(
            (j for j in range(n_steps) if float(sigmas[j]) <= t_thr),
            n_steps,
        )
        if adaptive:
            if ref_shift:
                # The reference split itself: what share of the schedule the
                # reference case spends coarse at this threshold. Derived from
                # the preset and delta, so it moves with both -- this is the
                # answer, not a bound on one.
                limit = math.ceil(n_steps * reference_coarse_fraction(t_thr, ref_shift))
                basis = "reference split"
            else:
                # Only for `custom`, where the model's schedule is unknown and
                # there is nothing to derive a share from. Shared across the
                # ladder: one transition gets MAX_COARSE_FRACTION, two split it.
                limit = int(n_steps * MAX_COARSE_FRACTION * (i + 1) / n_transitions)
                basis = "flat backstop, no ref_shift for this preset"
            if step_idx > limit:
                if log_fn is not None:
                    where = ("never reached in this schedule" if step_idx >= n_steps
                             else f"step {step_idx}")
                    log_fn(
                        f"adaptive delta: holding the {s_old:g}->{s_new:g} transition at "
                        f"step {limit} of {n_steps} ({basis}); this schedule's own "
                        f"crossing is at {where}. The gap means the schedule is more "
                        "compressed than the reference one -- a resolution-dependent "
                        "shift (ComfyUI ModelSamplingFlux / SwarmUI Sigma Shift) is the "
                        "usual cause."
                    )
                step_idx = limit
        if step_idx >= n_steps:
            break
        out.append((step_idx, s_old, s_new))
    return out


def _resolve_manual(
    sigmas: torch.Tensor, scales: List[float], manual_sigmas: List[float],
) -> List[Tuple[int, float, float]]:
    """Return transitions from user-specified sigma thresholds."""
    if len(scales) < 2:
        return []
    if len(manual_sigmas) != len(scales) - 1:
        raise ValueError(
            f"manual_sigmas has length {len(manual_sigmas)}, expected "
            f"{len(scales) - 1} (one threshold per transition in scales)."
        )
    out: List[Tuple[int, float, float]] = []
    n_steps = len(sigmas) - 1
    for s_old, s_new, thr in zip(scales[:-1], scales[1:], manual_sigmas):
        step_idx = next(
            (j for j in range(n_steps) if float(sigmas[j]) <= thr),
            n_steps,
        )
        if step_idx >= n_steps:
            break
        out.append((step_idx, s_old, s_new))
    return out


# =============================================================================
# Segmented sampling.
# =============================================================================

def _segment_callback(outer_cb, segment_start_idx: int):
    """Re-base callback step indices to the full schedule."""
    if outer_cb is None:
        return None
    def inner(d):
        d = dict(d)
        d["i"] = d.get("i", 0) + segment_start_idx
        outer_cb(d)
    return inner


@torch.no_grad()
def sample_speed_core(
    sampler_fn: Callable, model, x, sigmas, extra_args=None, callback=None, disable=None,
    *,
    transform: str = "dct",
    mode: str = "delta_optimal",
    scales: List[float] = None,
    delta: float = 0.01,
    spectrum_A: float = 203.615097,
    spectrum_beta: float = 1.915461,
    manual_sigmas: List[float] = None,
    seed: int = 0,
    adaptive_delta: bool = False,
    ref_latent: int = REFERENCE_LATENT,
    ref_shift: float = None,
    sampler_kwargs: dict = None,
    full_res_sampler_kwargs: dict = None,
    log_fn: Callable[[str], None] = None,
):
    """Run ``sampler_fn`` segment-by-segment with spectral expansion in between.

    ``sampler_fn`` is any k-diffusion style solver
    ``fn(model, x, sigmas, extra_args=..., callback=..., disable=..., **kw)``.
    ``sampler_kwargs`` are forwarded to every segment; ``full_res_sampler_kwargs``
    only to segments running at the final (1.0) scale — use it for objects tied
    to the full-resolution latent shape, e.g. a Brownian ``noise_sampler``.

    ``adaptive_delta`` holds the coarse/detail step split at the split the
    preset was calibrated for (``ref_latent``) instead of letting it grow with
    the canvas; see ``_resolve_transitions``. It only applies to
    ``delta_optimal`` mode -- ``manual`` thresholds are explicit sigmas and are
    already resolution-independent.
    """
    extra_args = {} if extra_args is None else extra_args
    sampler_kwargs = dict(sampler_kwargs or {})
    full_res_sampler_kwargs = dict(full_res_sampler_kwargs or {})

    H_full, W_full = x.shape[-2], x.shape[-1]

    if not scales or len(scales) < 2:
        return sampler_fn(model, x, sigmas, extra_args=extra_args,
                          callback=callback, disable=disable,
                          **{**sampler_kwargs, **full_res_sampler_kwargs})

    first_scale = scales[0]
    if mode == "delta_optimal":
        transitions = _resolve_transitions(
            sigmas, scales, delta, spectrum_A, spectrum_beta, H_full, W_full,
            adaptive=adaptive_delta, ref_latent=ref_latent, ref_shift=ref_shift,
            log_fn=log_fn,
        )
    elif mode == "manual":
        transitions = _resolve_manual(sigmas, scales, manual_sigmas or [])
    else:
        raise ValueError(f"mode must be delta_optimal|manual, got {mode!r}")

    if log_fn is not None:
        plan = ", ".join(
            f"step {idx}: {a:g}->{b:g} (sigma={float(sigmas[idx]):.4f})"
            for idx, a, b in transitions
        ) or "none"
        log_fn(f"transitions: {plan}")
        if transitions:
            n_steps = len(sigmas) - 1
            coarse = transitions[-1][0]
            log_fn(
                f"schedule split: {coarse}/{n_steps} steps coarse "
                f"({coarse / n_steps:.0%}), {n_steps - coarse} at full resolution"
            )
        if len(transitions) < len(scales) - 1:
            log_fn(
                "warning: not all transitions fit in the schedule; the final "
                f"latent stays at scale {scales[len(transitions)]:g} and the "
                "output will be smaller than requested. Increase steps, delta, "
                "or the manual sigma thresholds."
            )

    # DCT-truncate the incoming latent down to the coarsest scale.
    if first_scale < 1.0:
        x = _initial_dct_downscale(x, first_scale)

    sigmas = sigmas.clone()
    segment_starts = [0] + [t[0] for t in transitions]

    for seg_i, seg_start in enumerate(segment_starts):
        seg_end = transitions[seg_i][0] if seg_i < len(transitions) else len(sigmas) - 1
        seg_sigmas = sigmas[seg_start:seg_end + 1]
        if len(seg_sigmas) >= 2:
            cb = _segment_callback(callback, seg_start)
            seg_kwargs = dict(sampler_kwargs)
            if abs(scales[seg_i] - 1.0) < 1e-6:
                seg_kwargs.update(full_res_sampler_kwargs)
            x = sampler_fn(model, x, seg_sigmas, extra_args=extra_args,
                           callback=cb, disable=disable, **seg_kwargs)

        if seg_i >= len(transitions):
            break

        step_idx, s_i, s_next = transitions[seg_i]
        sigma_at_transition = float(sigmas[step_idx])
        x, t_tilde = _expand_and_align_torch(
            x, s_i, s_next, sigma_at_transition,
            transform=transform, seed=seed + (seg_i + 1) * 10000,
            H_full=H_full, W_full=W_full,
        )

        # Patch only the transition sigma, matching the reference inference loop.
        sigmas[step_idx] = float(t_tilde)

    return x
