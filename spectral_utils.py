"""Spectral expansion and transition scheduling utilities for SPEED.

Based on https://github.com/howardhx/speed
Extracted from the official speed/utils.py — used by speed_sampler.py.
"""
from __future__ import annotations

import functools
import math
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from scipy.fft import dctn, idctn


def power_spectrum(omega: float, A: float, beta: float) -> float:
    """Radial power-law spectrum ``P(omega) = A * |omega|**(-beta)``.

    Args:
    - omega: Radial spatial frequency.
    - A: Power-law amplitude (fitted per VAE model in ``configs.yaml``).
    - beta: Power-law decay exponent (fitted per VAE model in ``configs.yaml``).

    Returns:
    - The power-spectrum value ``P(omega)``.
    """
    return A * abs(omega) ** (-beta)


def activation_time(P_omega: float, delta: float) -> float:
    """Return the activation time for one radial frequency ``omega``.
    This matches Eq. 9 in the paper.

    Args:
    - P_omega: Power-spectrum value ``P(omega)`` at the frequency of interest.
    - delta: Noise-dominated tolerance; smaller ``delta`` delays activation.

    Returns:
    - The activation time ``t_omega`` in ``(0, 1)``.
    """
    if delta >= 1.0:
        raise ValueError(f"delta={delta} >= 1, but we assume the error threshold is < 1.")
    return 1.0 / (1.0 + math.sqrt(delta / (P_omega * (1.0 + P_omega - delta))))


def equivalent_delta(t_star: float, A: float, beta: float, omega: float) -> float:
    """Return the ``delta`` that puts ``activation_time`` at ``t_star`` for ``omega``.

    The closed-form inverse of ``activation_time``: with ``u = ((1-t)/t)**2``,
    ``delta = u*P*(1+P) / (1 + u*P)``. Used to report what an adaptive-delta run
    is doing in terms the ``delta`` input is expressed in.

    Args:
    - t_star: Target activation time in ``(0, 1)``.
    - A: Power-law amplitude.
    - beta: Power-law decay exponent.
    - omega: Radial spatial frequency the transition is evaluated at.

    Returns:
    - The equivalent ``delta``.
    """
    if not (0.0 < t_star < 1.0):
        raise ValueError(f"t_star must be in (0, 1); got {t_star}")
    P = power_spectrum(omega, A, beta)
    u = ((1.0 - t_star) / t_star) ** 2
    return u * P * (1.0 + P) / (1.0 + u * P)


def reference_coarse_fraction(t_star: float, shift: float) -> float:
    """Return the share of a schedule that sits above ``t_star``.

    For a schedule laid out uniformly in flow time ``t``, step ``j`` of ``n``
    sits at ``t = 1 - j/n`` and carries ``sigma = shift*t / (shift*t + 1 - t)``.
    Inverting that for the ``t`` where sigma crosses ``t_star`` gives
    ``T = t_star / (shift*(1 - t_star) + t_star)``, and the steps above the
    threshold are the ones with ``t > T`` -- a fraction ``1 - T``.

    This is what lets ``adaptive_delta`` express the reference split as a step
    *fraction* rather than a sigma. A sigma threshold is only equivalent to a
    fixed split while the schedule is fixed; once the shift moves with
    resolution the same sigma lands at a different step, and only the fraction
    survives the change.

    Args:
    - t_star: Transition sigma at the reference resolution.
    - shift: Effective schedule multiplier at the reference resolution (the
      ``exp(mu)`` value for ``PredictionFlux`` models, the raw multiplier for
      ``PredictionDiscreteFlow`` ones -- they are not the same units).

    Returns:
    - The coarse share in ``(0, 1)``.
    """
    if shift <= 0.0:
        raise ValueError(f"shift must be positive; got {shift}")
    if not (0.0 < t_star < 1.0):
        raise ValueError(f"t_star must be in (0, 1); got {t_star}")
    T = t_star / (shift * (1.0 - t_star) + t_star)
    return 1.0 - T


def delta_optimal_transitions(
    scales: Sequence[float],
    delta: float,
    A: float,
    beta: float,
    H: int,
    W: int,
) -> List[float]:
    """Return transition times for adjacent scales. This matches Eq. 10 from the paper.

    Args:
    - scales: Strictly increasing scale list ending at 1.0.
    - delta: Noise-dominated tolerance passed to ``activation_time``.
    - A: Power-law amplitude.
    - beta: Power-law decay exponent.
    - H: Full-resolution latent height (sets ``omega_max = min(H, W) / 2``).
    - W: Full-resolution latent width (sets ``omega_max = min(H, W) / 2``).

    Returns:
    - List of transition times ``t*_i`` (length ``len(scales) - 1``).
    """
    validate_scales(scales)
    omega_max = min(H, W) / 2.0
    transitions: List[float] = []
    for i in range(len(scales) - 1):
        omega_i = scales[i] * omega_max
        transitions.append(activation_time(power_spectrum(omega_i, A, beta), delta))
    return transitions


def align_timestep(t: float, r: float) -> float:
    """Return the aligned flow-matching time after spectral noise expansion.
    This matches Eq. 6 of the paper.

    Args:
    - t: Flow-matching time at the resolution transition.
    - r: Resolution scale ratio ``s_{i + 1} / s_i`` of the transition.

    Returns:
    - The aligned flow-matching time ``t_tilde``.
    """
    return t * kappa(t, r)


def kappa(t: float, r: float) -> float:
    """Return the state-rescaling factor after spectral noise expansion.
    This matches Eq. 5 of the paper.

    Args:
    - t: Flow-matching time at the resolution transition.
    - r: Scale ratio ``s_{i + 1} / s_i`` of the transition.

    Returns:
    - The state-rescaling factor ``kappa``.
    """
    return r / (1.0 + (r - 1.0) * t)


# =============================================================================
# On-device transforms (torch).
#
# An orthonormal DCT-II is a matrix product: ``D[k, i] = c_k cos(pi (i + 1/2)
# k / n)`` with ``c_0 = sqrt(1/n)`` and ``c_k = sqrt(2/n)`` otherwise, which is
# scipy's ``type=2, norm="ortho"``. ``dct2`` is ``D_h @ x @ D_w^T`` over the
# trailing two axes and matmul broadcasting covers every leading axis (batch,
# channel, frame) in one call, so nothing leaves the latent's device and
# there is no per-slice loop. The basis is cached per size and device; ``n``
# is a few hundred at most, so the ``n^3`` product is negligible. The numpy
# functions below stay as the reference the tests compare against.
# =============================================================================

@functools.lru_cache(maxsize=None)
def _dct_basis_cached(n: int, device: str) -> torch.Tensor:
    # Built in float64: the cosine argument reaches ~n*pi, where float32
    # would already lose the basis's orthogonality at latent sizes.
    i = torch.arange(n, dtype=torch.float64) + 0.5
    k = torch.arange(n, dtype=torch.float64).unsqueeze(1)
    basis = torch.cos((math.pi / n) * k * i)
    basis[0] *= math.sqrt(1.0 / n)
    if n > 1:
        basis[1:] *= math.sqrt(2.0 / n)
    return basis.to(dtype=torch.float32, device=torch.device(device))


def dct_basis(n: int, device) -> torch.Tensor:
    """Return the cached orthonormal DCT-II basis ``(n, n)`` on ``device``."""
    return _dct_basis_cached(int(n), str(device))


def dct2(x: torch.Tensor) -> torch.Tensor:
    """Orthonormal 2D DCT-II over the trailing two axes, in float32."""
    x = x.float()
    return dct_basis(x.shape[-2], x.device) @ x @ dct_basis(x.shape[-1], x.device).T


def idct2(coeffs: torch.Tensor) -> torch.Tensor:
    """Inverse of ``dct2``."""
    coeffs = coeffs.float()
    return dct_basis(coeffs.shape[-2], coeffs.device).T @ coeffs @ dct_basis(coeffs.shape[-1], coeffs.device)


def dct_truncate(x: torch.Tensor, H_lo: int, W_lo: int) -> torch.Tensor:
    """Keep the ``H_lo x W_lo`` low band of ``x``'s DCT, on an ``H_lo x W_lo`` grid.

    Per coefficient this is the identity, so white noise keeps its variance.
    Smooth content does not keep its level: the DC coefficient is
    ``mean * sqrt(H*W)``, so a constant comes back scaled by
    ``sqrt(H*W / (H_lo*W_lo))``.
    """
    H, W = x.shape[-2], x.shape[-1]
    if H_lo > H or W_lo > W or H_lo < 1 or W_lo < 1:
        raise ValueError(f"DCT truncate: target ({H_lo}, {W_lo}) must fit inside ({H}, {W})")
    return idct2(dct2(x)[..., :H_lo, :W_lo])


def _fresh_normal(shape: Tuple[int, ...], seed: int, device) -> torch.Tensor:
    """Standard normal noise from the numpy generator, so a seed gives the
    same fill on every device and the same values the numpy path drew."""
    rng = np.random.default_rng(seed)
    return torch.from_numpy(rng.standard_normal(shape).astype(np.float32)).to(device)


def dct_expand(
    x: torch.Tensor, target_hw: Tuple[int, int], t: float, seed: int,
    fill: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """DCT spectral noise expansion on the latent's device.

    Args:
    - x: Source tensor; trailing two axes are the spatial grid to expand.
    - target_hw: Target ``(height, width)``.
    - t: Noise amplitude for the new high-frequency coefficients.
    - seed: Seed for the fresh draw, when ``fill`` is not given.
    - fill: Optional DCT coefficients of a full-resolution unit-variance noise
      field, same leading axes as ``x``; its ``[:H_tgt, :W_tgt]`` block times
      ``t`` fills the new band instead of a fresh draw (coupled noise).

    Returns:
    - The expanded tensor at ``target_hw`` (float32).
    """
    H_tgt, W_tgt = int(target_hw[0]), int(target_hw[1])
    H_src, W_src = x.shape[-2], x.shape[-1]
    if H_tgt < H_src or W_tgt < W_src:
        raise ValueError(
            f"DCT expand: cannot expand to target {target_hw} smaller than "
            f"source ({H_src}, {W_src})."
        )
    coeffs = dct2(x)
    if fill is not None:
        big = fill[..., :H_tgt, :W_tgt].to(device=x.device, dtype=torch.float32) * float(t)
    else:
        big = float(t) * _fresh_normal(tuple(x.shape[:-2]) + (H_tgt, W_tgt), seed, x.device)
    big[..., :H_src, :W_src] = coeffs
    return idct2(big)


def fft_expand(
    x: torch.Tensor, target_hw: Tuple[int, int], t: float, seed: int,
    fill: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """FFT spectral noise expansion on the latent's device.

    ``fill`` is the centred (``fftshift``) orthonormal FFT of a full-resolution
    unit-variance noise field; its centre block of the target size times ``t``
    fills the new band. Without it the draw matches ``_fft_expand_np`` for the
    same seed.
    """
    H_tgt, W_tgt = int(target_hw[0]), int(target_hw[1])
    H_src, W_src = x.shape[-2], x.shape[-1]
    if H_tgt < H_src or W_tgt < W_src:
        raise ValueError(
            f"FFT expand: cannot expand to target {target_hw} smaller than "
            f"source ({H_src}, {W_src})."
        )
    pad_h, pad_w = (H_tgt - H_src) // 2, (W_tgt - W_src) // 2
    X_src = torch.fft.fftshift(torch.fft.fft2(x.float(), norm="ortho"), dim=(-2, -1))
    if fill is not None:
        H_full, W_full = fill.shape[-2], fill.shape[-1]
        off_h, off_w = (H_full - H_tgt) // 2, (W_full - W_tgt) // 2
        X_big = fill[..., off_h:off_h + H_tgt, off_w:off_w + W_tgt].to(device=x.device) * float(t)
    else:
        # Per slice, real then imaginary, in the numpy path's draw order.
        rng = np.random.default_rng(seed)
        lead = tuple(x.shape[:-2])
        noise = np.empty(lead + (H_tgt, W_tgt), dtype=np.complex64)
        for idx in np.ndindex(*lead):
            nr = rng.standard_normal((H_tgt, W_tgt)).astype(np.float32)
            ni = rng.standard_normal((H_tgt, W_tgt)).astype(np.float32)
            noise[idx] = np.fft.fftshift(float(t) * (nr + 1j * ni) / np.sqrt(2.0))
        X_big = torch.from_numpy(noise).to(x.device)
    X_big[..., pad_h:pad_h + H_src, pad_w:pad_w + W_src] = X_src
    return torch.fft.ifft2(torch.fft.ifftshift(X_big, dim=(-2, -1)), norm="ortho").real


# =============================================================================
# Reference implementations (numpy). Kept as the oracle for the tests.
# =============================================================================

def _dct_expand_np(
    x_np: np.ndarray, target_hw: Tuple[int, int], t: float, seed: int,
) -> np.ndarray:
    """DCT spectral noise expansion.

    Args:
    - x_np: Source array; trailing two axes are the spatial grid to expand.
    - target_hw: Target ``(height, width)`` of the expanded grid.
    - t: Noise amplitude for the high-frequency coefficients.
    - seed: Seed for the per-call random generator.

    Returns:
    - The expanded array at ``target_hw`` (float32, same leading axes as ``x_np``).
    """
    H_tgt, W_tgt = target_hw
    H_src, W_src = x_np.shape[-2], x_np.shape[-1]
    if H_tgt < H_src or W_tgt < W_src:
        raise ValueError(
            f"DCT expand: cannot expand to target {target_hw} smaller than "
            f"source ({H_src}, {W_src})."
        )
    rng = np.random.default_rng(seed)
    out = np.empty(x_np.shape[:-2] + (H_tgt, W_tgt), dtype=np.float32)
    for idx in np.ndindex(*x_np.shape[:-2]):
        coeffs_src = dctn(x_np[idx], type=2, norm="ortho")
        big = t * rng.standard_normal((H_tgt, W_tgt)).astype(np.float32)
        big[:H_src, :W_src] = coeffs_src
        out[idx] = idctn(big, type=2, norm="ortho").astype(np.float32)
    return out


def _dwt_expand_np(x_np: np.ndarray, t: float, seed: int) -> np.ndarray:
    """Haar wavelet spectral noise expansion. The target H, W is automatically
    two times the source H, W.

    Args:
    - x_np: Source array treated as the LL band; trailing two axes are spatial.
    - t: Noise amplitude for the LH/HL/HH detail bands.
    - seed: Seed for the per-call random generator.

    Returns:
    - The expanded array at twice the source resolution (float32).
    """
    try:
        import pywt
    except ImportError as e:
        raise ImportError(
            "transform=dwt requires the PyWavelets package; install it with "
            "`pip install PyWavelets`, or use transform=dct / transform=fft."
        ) from e
    H_src, W_src = x_np.shape[-2], x_np.shape[-1]
    H_tgt, W_tgt = H_src * 2, W_src * 2
    rng = np.random.default_rng(seed)
    out = np.empty(x_np.shape[:-2] + (H_tgt, W_tgt), dtype=np.float32)
    for idx in np.ndindex(*x_np.shape[:-2]):
        LL = x_np[idx]
        LH = t * rng.standard_normal(LL.shape).astype(np.float32)
        HL = t * rng.standard_normal(LL.shape).astype(np.float32)
        HH = t * rng.standard_normal(LL.shape).astype(np.float32)
        out[idx] = pywt.waverec2(
            [LL, (LH, HL, HH)], "haar", mode="periodization"
        ).astype(np.float32)
    return out


def _fft_expand_np(
    x_np: np.ndarray, target_hw: Tuple[int, int], t: float, seed: int,
) -> np.ndarray:
    """FFT spectral noise expansion.

    Args:
    - x_np: Source array; trailing two axes are the spatial grid to expand.
    - target_hw: Target ``(height, width)`` of the expanded grid.
    - t: Noise amplitude for the outer (high-frequency) spectrum.
    - seed: Seed for the per-call random generator.

    Returns:
    - The expanded array at ``target_hw`` (float32, same leading axes as ``x_np``).
    """
    H_tgt, W_tgt = target_hw
    H_src, W_src = x_np.shape[-2], x_np.shape[-1]
    if H_tgt < H_src or W_tgt < W_src:
        raise ValueError(
            f"FFT expand: cannot expand to target {target_hw} smaller than "
            f"source ({H_src}, {W_src})."
        )
    rng = np.random.default_rng(seed)
    pad_h, pad_w = (H_tgt - H_src) // 2, (W_tgt - W_src) // 2
    out = np.empty(x_np.shape[:-2] + (H_tgt, W_tgt), dtype=np.float32)
    for idx in np.ndindex(*x_np.shape[:-2]):
        X_src = np.fft.fftshift(np.fft.fft2(x_np[idx], norm="ortho"))
        nr = rng.standard_normal((H_tgt, W_tgt)).astype(np.float32)
        ni = rng.standard_normal((H_tgt, W_tgt)).astype(np.float32)
        X_big = np.fft.fftshift(t * (nr + 1j * ni) / np.sqrt(2.0))
        X_big[pad_h:pad_h + H_src, pad_w:pad_w + W_src] = X_src
        out[idx] = np.fft.ifft2(np.fft.ifftshift(X_big), norm="ortho").real.astype(np.float32)
    return out


def validate_scales(scales: Sequence[float]) -> None:
    """Validate a strictly increasing resolution scale list ending at 1.0.

    Args:
    - scales: Scale list to validate; each value in ``(0, 1]``, strictly
      increasing, ending at ``1.0``.

    Returns:
    - None; raises ``ValueError`` if the scales are invalid.
    """
    if len(scales) == 0:
        raise ValueError("list of resolution scales is empty; supply at least one value.")
    if any(s <= 0.0 or s > 1.0 for s in scales):
        raise ValueError(f"every scale must be in (0, 1]; got {list(scales)}")
    if abs(scales[-1] - 1.0) > 1e-6:
        raise ValueError(f"last scale must equal 1.0 (full resolution); got {scales[-1]}")
    for a, b in zip(scales[:-1], scales[1:]):
        if not (a < b):
            raise ValueError(f"scales must be strictly increasing; got {list(scales)}")