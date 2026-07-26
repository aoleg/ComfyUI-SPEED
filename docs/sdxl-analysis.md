# Why SPEED goes soft on SDXL

Status: **analysis only, nothing here is implemented.** Written to decide whether SDXL
support is worth building and what it would have to change. Numbers below are for
SDXL at 1024x1024 (128x128 latent), `scales=0.5,1.0`, Karras 30 steps
(`sigma_max=14.6146`, `sigma_min=0.0292`, `rho=7`).

The short version: SPEED's transition math is derived for the flow-matching mixture
`x_t = (1-t)x0 + t*eps` with `t` in `(0, 1]`. Two separate places assume that, and both
silently produce a wrong-but-finite answer on an eps-prediction model, where
`x = x0 + sigma*eps` and sigma runs to 14.6. One makes the transition far too late, the
other halves the image contrast at the transition. Softness is what both of them look like.

---

## 1. The transition threshold is compared in the wrong sigma space

`activation_time` (Eq. 9) returns a **flow time** in `(0, 1)`. `_resolve_transitions`
compares it directly against the schedule's sigmas:

```python
step_idx = next((j for j in range(n_steps) if float(sigmas[j]) <= t_thr), n_steps)
```

On a flow model that is right, because sigma *is* the flow time. On an eps model it is a
category error. The equivalent noise level is the one with the same SNR — the criterion is
"signal power `(1-t)^2 P(w)` versus noise power `t^2`", and the eps form is "`P(w)` versus
`sigma^2`" — so

```
sigma* = t* / (1 - t*)
```

| preset | `t*` (delta=0.01) | compared as-is | should be `sigma*` |
|---|---|---|---|
| `wan21` | 0.6941 | 0.6941 | 2.269 |
| `flux` | 0.8527 | 0.8527 | 5.789 |
| `krea-2` | 0.9000 | 0.9000 | 9.000 |
| `z-image` | 0.9400 | 0.9400 | 15.667 (above `sigma_max`: expand at step 0) |

Where that lands on Karras 30 (`sigma[16] = 0.9357`, `sigma[17] = 0.7558`):

| threshold | transition | steps at half res |
|---|---|---|
| 0.6941 as-is | step 18 | 60% |
| 0.8527 as-is | step 17 | 57% |
| 0.9400 as-is | step 16 | 53% |
| 5.789 (correct for `flux`) | step 7 | 23% |

**This is why no preset helps.** Every delta-optimal threshold is below 1.0 by
construction (`activation_time` returns a value in `(0,1)` for any `A`, `beta`, `delta`),
and the first Karras sigma below 1.0 is step 16 of 30. So the entire preset x delta control
surface can only place the transition between steps 16 and 20 — the model spends
55-60% of the run at half resolution no matter what, and the last high-frequency third of
the schedule is the only part that ever sees the full grid. By SPEED's own criterion the
band above the coarse Nyquist became signal-dominated around step 7; everything sampled
at half resolution after that is real signal being discarded, and discarded
high-frequency signal is exactly what "soft" means.

Fix: convert the threshold when the predictor is not flow-matching. Detect it explicitly,
per `knowledge_sigmas.md` §2 — `predictor.prediction_type == "const"` in Forge Neo,
`model_sampling` class in ComfyUI — and fall back to a `sigmas[0] > 1.5` heuristic only if
the predictor is unreachable. The Forge script already logs a note on that heuristic; the
note should become a conversion.

## 2. The state rescale `kappa` is derived for the flow mixture only

`_expand_and_align_torch` copies the source DCT coefficients into the corner of a larger
ortho-DCT grid. That preserves total coefficient energy, so per-pixel amplitude drops by
`r`. Both the signal and the noise shrink by `r`; the injected high-frequency band is
generated at amplitude `t`, so after the inverse transform:

```
signal  x0 * (1-t) / r        noise  variance t^2   (low band t^2/r^2 + new band t^2(1-1/r^2))
```

The noise is already at the right level; only the signal is short by `r`. The flow
parameterisation can repair that with one global factor because its two coefficients must
sum to 1 — solving `1 - t~ = (1-t) kappa / r` and `t~ = t kappa` gives exactly
`kappa = r / (1 + (r-1) t)` and `t~ = t kappa`, which is what the code implements. It is
correct, and self-consistent, *for flow*.

VE/eps has no such budget: the signal coefficient is fixed at 1 and cannot be traded
against the noise. Applying `kappa` there scales signal and noise together, so the
declared sigma still matches the actual noise, but the image content is left at

```
kappa / r  =  1 / (1 + (r-1) sigma)
```

of its correct amplitude:

| sigma at the transition | `kappa` | signal retained | sigma patched to |
|---|---|---|---|
| 5.79 | 0.295 | **15%** | 1.705 |
| 2.00 | 0.667 | **33%** | 1.333 |
| 0.853 | 1.079 | **54%** | 0.921 |
| 0.300 | 1.538 | 77% | 0.462 |

At the sigma the transition currently fires (~0.85) the latent handed to the second
segment holds a half-contrast picture in a full dose of noise, and the sigma is *raised*
from 0.853 to 0.921 on top of that. A denoiser given a signal at half the SNR its sigma
implies returns a flat, low-detail estimate. Note this gets worse, not better, once §1 is
fixed — at sigma 5.79 only 15% of the signal survives — so §1 cannot be shipped without
§2.

Two ways to repair it, both needing the denoised estimate `x0_hat` (free: k-diffusion
samplers hand it to the callback every step):

* **Amplitude top-up, keeps the noise realisation.** Expand as today but with
  `kappa = 1`, then add back the signal the expansion diluted:

  ```
  x_new = expand(x) + c * (r - 1) * expand_noiseless(x0_hat)      # sigma unchanged
  ```

  where `c` is the signal coefficient of the parameterisation — `1` for eps/VE, `(1-t)`
  for flow. `expand(x)` contributes `c*x0_up/r`, the second term contributes
  `c*x0_up*(r-1)/r`, and the two sum to `c*x0_up` with the noise untouched at its correct
  level. No timestep realignment is needed, so this would retire `kappa` and
  `align_timestep` for both families rather than special-casing eps.
* **Expand-and-renoise.** `x_new = c * r * expand_noiseless(x0_hat) + sigma * eps_full`.
  Exact in both parameterisations and simpler, but it discards the correlation with the
  existing noise, i.e. it behaves like one ancestral step at the transition.

## 3. Even with 1 and 2 fixed, a conv UNet at 0.5 scale is out of distribution

This is the part `knowledge_raunet.md` is useful for — not its code, its premise. RAUNet
exists because SDXL's UNet has a **resolution pyramid with a trained receptive field**
(`§9`: 9 input blocks, Downsample at 3 and 6, cross-attention at input 4/5/7/8), and
running it away from 1024 breaks the relationship between that receptive field and image
content. HiDiffusion/RAUNet fix the *above*-native case by rewriting Downsample geometry
so the deep blocks keep their trained field. SPEED creates the mirror case: at
`scales=0.5,1.0` the first half of the run is SDXL at 512x512, where its deepest level
sees an 8x8 grid instead of 16x16 — the regime SDXL is well known to compose badly in.

A DiT does not have this problem in the same way: Flux/Wan/Krea 2/Z-Image are flat
transformer stacks over RoPE-indexed patches (`knowledge_raunet.md` §8 notes Krea 2's
`SingleStreamDiT` has no pyramid at all), so a shorter sequence is a smaller picture
rather than a differently-shaped model.

There is a second, SDXL-specific mismatch on top of the architectural one: SDXL's **size
conditioning** (`backend/diffusion_engine/sdxl.py:90` — `original_size`, `crop`,
`target_size`) is embedded into the conditioning once, from the requested width/height,
and never updated. During the coarse segment the model is told it is producing 1024x1024
while being handed a 64x64 latent. Flux and friends carry no such signal.

Consequences for a design:

* Prefer a **milder first scale** on conv UNets — `0.75,1.0` (768px, inside SDXL's
  competent range) rather than `0.5,1.0`. DCT handles the non-dyadic ratio already.
* If `0.5` is wanted, the honest version is a RAUNet-shaped idea rather than a preset:
  during the coarse segment only, change the Downsample geometry so the deep blocks keep
  their 1024-native field (borrow the layer and swap `stride`/`dilation` around the call,
  `knowledge_raunet.md` §5; do not rebuild the conv). That is a much bigger project than
  SPEED currently is, and it must be gated per pass and by runtime latent shape
  (`§7.1`, `§7.2`).
* Re-derive the size conditioning per segment, or accept the mismatch and document it.

## 4. A measured SDXL spectrum is needed, but it is the last step, not the first

The presets are fitted on 16-channel latents. The SD 1.x/SDXL VAE is 4-channel at the same
8x compression, so each latent channel carries proportionally more high-frequency content —
which pushes `t*` *up*, i.e. the transition even earlier than the 5.79 above. Fitting
`A`/`beta` on SDXL latents is cheap and worth doing, but on its own it moves the transition
by a step or two inside a window that is already 10 steps too late. Order of work: §1, then
§2, then the fit, then §3.

## 5. Expected payoff, stated up front

With §1 and §2 done, the honest ceiling for SDXL at 1024 is roughly **23% of steps at
quarter cost, i.e. ~1.2x**, against ~1.4x for the flow models — because on an eps schedule
the high-frequency band becomes signal-dominated much earlier in the run. If that is not
worth the work, the alternative is to stop pretending: detect a non-flow predictor and
refuse with one clear line, the way `knowledge_raunet.md` §8 recommends for architectures
that cannot support an effect.

## 6. Interactions to check before any of this ships

* **ControlNet is silently dropped at coarse resolution.** `backend/nn/unet.py:46`
  `apply_control` does `h += ctrl` inside a bare `try/except` and prints one warning line
  on a shape mismatch (`knowledge_raunet.md` §6.3). Every SDXL run with a ControlNet would
  lose its hint for the whole coarse segment. Needs an interpolate-to-`h` shim or an
  explicit skip.
* **Hires-fix.** Already gated behind a checkbox in the Forge script; keep it off by
  default for eps models — the first pass is the one at native resolution.
* **Samplers that re-enter the model at another resolution** (`Euler Dy` / `SMEA Dy`,
  `knowledge_raunet.md` §7.2) interact badly with a segment that has already changed the
  latent shape.
* **Inpainting** is skipped already, for the same reason it must be.

## 7. What to measure first

Cheap, and it settles §1 and §2 without a single image:

1. Log the SDXL schedule, the resolved transition step, and the sigma there — one line,
   the highest-value thing per `knowledge_sigmas.md` §6.
2. Assert the round trip: expand a known latent at `kappa=1` and measure the per-pixel std
   of the signal band before and after. It should come out at `1/r`; that is the bug in §2
   in one number.
3. Then, and only then, A/B the corrected threshold at `0.75,1.0` against baseline.
