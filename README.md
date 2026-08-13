# ComfyUI-SPEED

ComfyUI custom node, **Forge Neo (sd-webui-forge-classic) extension**, and **SwarmUI extension** for **SPEED** — Spectral Progressive Diffusion for faster sampling. Progressively expands the latent resolution during denoising, reducing computation while preserving visual quality.

**Official code:** https://github.com/howardhx/speed

Key references:

- Project page: https://howardxiao.ca/speed/
- Paper (PDF): https://howardxiao.ca/speed/paper/paper.pdf
- arXiv: https://arxiv.org/abs/2605.18736

Workflow:

![Workflow](images/workflow.png)

## Speed comparison (FLUX preset, default parameters)

| SPEED sampler (this node)                                                 | Baseline (euler sampler) |
| ------------------------------------------------------------------------- | ------------------------ |
| `mode=delta_optimal` `model_preset=flux` `scales=0.5,1.0` `delta=0.01` `transform=dct` `base_sampler=euler`<br><br>![SPEED](images/anima_speed.png)<br><br>**20s**<br>**1.33x faster** | ![Original](images/anima_original.png)<br><br>**26.5s**<br>**1.00x** |

## Usage (ComfyUI)

1. Place this folder under your ComfyUI `custom_nodes` directory, then restart ComfyUI.
2. Connect the **Sampler SPEED (Spectral Progressive Diffusion)** output to **SamplerCustomAdvanced**.

## Usage (Forge Neo / sd-webui-forge-classic)

The same repo doubles as a webui extension — the shared math lives in `speed_core.py`, the webui integration in `scripts/speed_forge.py`.

1. Place (or `git clone`) this folder under the webui `extensions/` directory and restart the webui.
2. Open the **SPEED (Spectral Progressive Diffusion)** accordion under the generation parameters and enable it.
3. Generate. The sampler selected in the main **Sampling method** dropdown is used as the base solver; SPEED segments its sigma schedule and expands the latent at each transition.

Notes for the webui version:

- Designed for **flow-matching models** (Flux, Wan, Krea 2, Z-Image, …) whose sigmas are in `(0, 1]`. eps-prediction models (SD1.x/SDXL) run but are unvalidated and currently produce soft results — `docs/sdxl-analysis.md` explains why. In `manual` mode thresholds above 1.0 are accepted for such schedules.
- **Inpainting** is automatically skipped (masks and inpaint image conditioning are tied to the full latent resolution).
- The **hires-fix pass** is untouched unless "Apply to hires-fix pass" is checked.
- Brownian-noise samplers (DPM++ SDE family) work, but their pre-built noise sampler is only used on the full-resolution segment; coarse segments fall back to the sampler's default noise.
- The `dwt` transform needs the optional `PyWavelets` package (`pip install PyWavelets` into the webui venv); `dct` (default) and `fft` have no extra dependencies.
- If a transition threshold is never reached (too few steps / too small `delta`), the console warns and the output stays at the coarse resolution — increase steps or `delta`.
- Settings are recorded in the image infotext under the `SPEED:` key.
- The webui's **Mode** dropdown has a third option, `neo_shift` (default), not present in the ComfyUI node or the SwarmUI extension — see [neo_shift mode](#neo_shift-mode-forge-neo-only).

## Usage (SwarmUI)

The same repo also works as a SwarmUI extension — the C# integration lives in `SPEEDSamplerExtension.cs`, and SwarmUI's portable ComfyUI backend runs the `SamplerSPEED` node from this repo.

1. `git clone` this repo into SwarmUI's `src/Extensions/` directory (e.g. `src/Extensions/ComfyUI-SPEED/`), then run SwarmUI's `update` script (or launch with `launch-dev`) so the extension compiles.
2. In the Generate tab, enable **Display Advanced Options**, find the **SPEED** parameter group, and toggle it on. If the ComfyUI-SPEED node pack isn't installed on the backend yet, click the **Install SPEED** button inside the group (this clones this same repo into the backend's `DLNodes` and restarts it).
3. Generate. The main **Sampler** parameter is used as the base solver.

> **The extension and the node pack must come from the same repo.** They are two separate installs: the C# extension in `src/Extensions/ComfyUI-SPEED`, and the Python node pack in the backend's `.../ComfyUIBackend/DLNodes/ComfyUI-SPEED`. Rebuilding SwarmUI updates only the first. If they diverge, ComfyUI builds the node's arguments from its own `INPUT_TYPES` and **silently drops** anything the installed node doesn't declare — so newer parameters appear in the UI, change nothing, and report no error. The extension now warns in the SwarmUI log when it detects this, but to check or fix it directly:
>
> ```bash
> git -C <SwarmUI>/src/BuiltinExtensions/ComfyUIBackend/DLNodes/ComfyUI-SPEED remote -v
> ```
>
> If that is not this repo, re-point it (`git remote set-url origin <this repo>` then `git pull`) or delete the folder and use **Install SPEED** again. SwarmUI skips cloning when the folder already exists and `git pull`s it on every startup, so a wrong remote is sticky and keeps re-asserting itself.

Notes for the SwarmUI version:

- When SPEED is enabled, the main sampling node is rebuilt as a `SamplerCustomAdvanced` + `SamplerSPEED` graph. Swarm-specific sampler features — **Variation Seed** and tiled sampling — do not apply while SPEED is on, and live previews may behave differently.
- Swarm-custom schedulers (`align_your_steps`, `flux2`, `ltxv`, `ideogram4`, ...) can't be rebuilt from stock Comfy nodes; SPEED falls back to `simple` for those (with a log warning). `turbo`, `karras`, and all standard Comfy schedulers are reproduced faithfully.
- **Masked generations** (inpainting / regional masks) are automatically skipped, and the **Refiner stage** is untouched.
- The base **Sampler** must be a k-diffusion sampler (Euler, DPM++ 2M, etc.). Exotic non-k-diffusion samplers (`ddim`, `uni_pc`, `res_multistep`) are rejected by the node with a clear error — pick another sampler.
- The SPEED parameters are recorded in the image metadata like all Swarm params.

## Inputs

| Input | Type | Default | Description |
|---|---|---|---|
| `base_sampler` | combo | `euler` | Underlying ODE solver. Any `comfy.k_diffusion.sampling` sampler supported. |
| `transform` | combo | `dct` | Spectral basis for expansion: `dct` (any ratio), `dwt` (2× only), `fft` (any ratio). |
| `mode` | combo | `delta_optimal` | `delta_optimal` computes transitions from the power-spectrum preset. `manual` uses user-specified sigma thresholds. |
| `model_preset` | combo | `flux` | Power-spectrum preset: `flux`, `wan21`, `krea-2`, `krea-2-raw`, `z-image`, `flux2`, `anima`, or `custom` (use `spectrum_A` / `spectrum_beta`). See [Model presets](#model-presets). |
| `scales` | string | `0.5,1.0` | Comma-separated resolution fractions ending at `1.0`. e.g. `0.25,0.5,1.0`. |
| `adaptive_delta` | bool | `true` | Hold the coarse/detail step split at the preset's 1024x1024 calibration instead of letting the coarse pass grow with resolution. See [Adaptive delta](#adaptive-delta-on-by-default). |
| `delta` | float | `0.01` | Noise-dominated tolerance (Eq. 9). Smaller values delay transitions. With `adaptive_delta` on, this is the value at the 1024x1024 reference. |
| `manual_sigmas` | string | `0.85` | Comma-separated sigma thresholds (one per transition). Used in `manual` mode. |
| `spectrum_A` | float | `203.615` | Power-law amplitude (used when `model_preset=custom`). |
| `spectrum_beta` | float | `1.915` | Power-law decay exponent (used when `model_preset=custom`). |
| `seed` | int | `0` | Seed for spectral-noise padding at each transition. |
| `log` | bool | `true` | Print the resolved transition step(s) and sigma(s) to the console every run, e.g. `[SPEED] transitions: step 8: 0.5->1 (sigma=0.8473)`. The fastest way to check a preset/delta landed where you expect at your actual resolution and step count. |

### delta_optimal mode (recommended)

Pick the `model_preset` that matches your checkpoint and adjust `scales` and `delta`. Transition timing is computed automatically from the latent power spectrum using Eq. 9 and Eq. 10 of the paper, and [adaptive delta](#adaptive-delta-on-by-default) keeps that timing consistent as you change resolution.

### Model presets

The preset supplies the power spectrum `P(w) = A * w^-beta`; SPEED evaluates it at the coarse grid's Nyquist frequency and converts it into the sigma at which the latent is expanded. The last column is that sigma for the reference case — 1024x1024, `scales=0.5,1.0`, `delta=0.01`.

| Preset | Latent space | Intended for | sigma\* | basis |
|---|---|---|---|---|
| `flux` | Flux VAE | Flux.1 | 0.9680 | **measured** |
| `flux2` | Flux VAE (borrowed) | FLUX.2 [dev] / Klein, 40-50 steps | 0.9680 † | **measured** |
| `wan21` | Wan 2.1 VAE | Wan 2.1 video | 0.6941 | upstream published fit |
| `krea-2` | Wan 2.1 / Qwen-Image | Few-step Krea 2: the Turbo checkpoint at ~8 steps, or the base model + Turbo LoRA at ~20 steps | 0.9000 | spectrum-derived, untested |
| `krea-2-raw` | Wan 2.1 / Qwen-Image | Undistilled Krea 2 base model, 40-50 steps with CFG | 0.8300 | spectrum-derived, untested |
| `z-image` | Flux VAE | Z-Image Turbo, 8-9 steps | 0.9400 | spectrum-derived, untested |
| `anima` | Wan 2.1 / Qwen-Image | Anima, 32 steps with CFG | 0.9600 | spectrum-derived, untested |
| `custom` | — | Your own `spectrum_A` / `spectrum_beta` | — | — |

† FLUX.2's sampler latent is 1/16 scale, so its sigma\* is quoted at a 64x64 reference grid; every other preset uses 128x128. Same 1024x1024 image either way.

#### The quality boundary is a sigma

`flux` and `flux2` are calibrated from A/B output rather than from the spectrum, and **`flux` no longer ships the published fit** (`A 203.615097`, sigma\* 0.8527 — restorable via `custom`). The measurement that moved it: on Flux.1 dev at 1024x1024, transitioning at sigma 0.9389 renders license-plate text as wrong letters, while 0.9611 renders it correctly. The published threshold of 0.8527 is far below both.

What made this findable is that the boundary is a **sigma**, and it holds across models, resolutions, and schedules differing by 7.6×:

| model | res | coarse/40 | sigma at transition | result |
|---|---|---|---|---|
| FLUX.2 | 1536 | 15 | 0.9915 | indistinguishable from off |
| Flux.1 | 1536 | 15 | 0.9915 | text fine |
| Flux.1 | 1024 | 12 | 0.9611 | text correct |
| Flux.1 | 1024 | 15 | 0.9389 | **text broken** |
| FLUX.2 | 1536 | 29 | 0.9362 | sharp, small details missing |
| FLUX.2 | 1536 | 33 | 0.8628 | coherent but visibly simpler |
| FLUX.2 | 1536 | 35 | 0.7762 | **broken** |

Everything at sigma ≥ 0.96 passes. Note the **step counts do not transfer** — Flux.1 needed 28 full-resolution steps at 1024px but only 25 at 1536px, because fine detail lands on more pixels as the canvas grows. Only the sigma is invariant, which is why these presets pin the threshold rather than a step budget. `0.968` reproduces both measured optima (11–12 of 40 coarse at 1024px, 25 of 40 at 1536px).

A consequence worth knowing: under a fixed sigma threshold **SPEED's payoff grows with resolution**, because a larger canvas runs a more compressed schedule and so puts more steps above the threshold. Measured with `flux` at 40 steps: ~1.15× at 1024px on a stock schedule, ~1.9× at 1536px.

The remaining five presets are untouched pending their own A/B. `beta` for `krea-2`/`krea-2-raw`/`z-image`/`anima` is taken from the measured fit for the same VAE latent space, and `A` solved so the default `delta` places the transition where their step budget wants it; `wan21` is the upstream fit as published. `tests/test_math.py` asserts every sigma\* above.

`anima` is the best-founded of the untested group, because all three of its numbers are read from the model's shipped config rather than inferred from lineage. Anima is a Cosmos-Predict2-class DiT (`CosmosTransformer3DModel`, 28 blocks at 2048 channels) re-textencoded onto Qwen3-0.6B, and it encodes with `AutoencoderKLQwenImage` — 16 channels at 8x, with `latents_mean`/`latents_std` byte-identical to Wan 2.1's, so `beta` carries over on the same basis `krea-2` uses. Two traps it *doesn't* fall into: its `patch_spatial=2` packing happens inside the DiT's `x_embedder`, downstream of the sampler, so `ref_latent` stays 128 (FLUX.2 packs in the latent format instead, which is why that one is 64 — same patch size, opposite answer); and it is a `PredictionDiscreteFlow` model, so its `ref_shift` of 3.0 is a **raw multiplier** in the `z-image`/`wan21` units, not the `exp(mu)` the `flux`/`krea-2` presets use.

sigma\* 0.9600 is picked from where the Forge Neo default path lands rather than from the step budget alone: that webui defaults to [`neo_shift`](#neo_shift-mode-forge-neo-only) with a 1.03 divisor, so the threshold actually used is `0.9600 / 1.03 = 0.9320`, clearing the measured 0.9249 floor with margin. Anima also declares `use_dynamic_shifting: false`, which no other preset here does — its schedule shift is flat at every resolution on **every** backend, so unlike `flux` its payoff does not grow with the canvas. At its 32-step preset, `scales=0.5,1.0`, adaptive on, it transitions at 6/32 for roughly 1.16× at both 1024px and 1536px.

`flux2` is weaker than that, and is flagged accordingly. FLUX.2's VAE encodes at 8x into 32 channels and then packs 2x2 spatially into the channel axis (`AutoencoderKLFlux2.encode`, `patch_size: [2, 2]`), so the tensor the sampler — and therefore SPEED — actually sees is 128 channels at **1/16 scale**: a 1024x1024 image is a 64x64 grid, not the 128x128 every other preset here works on. Its sigma\* is calibrated at that geometry, which is why its `A` looks small next to `flux`'s despite targeting a *later* transition. Its `beta` is Flux.1's only because that's the closest available analog (same lineage), not because the latents match in shape or channel count — treat it as a starting point and refine with `custom` if you can A/B it. What *is* source-backed: FLUX.2 Klein 4B/9B both declare `shift = 2.02` in ComfyUI's `supported_models.py` (log-space, like Flux.1/Krea 2 — an effective multiplier of `exp(2.02) = 7.54`); no separate entry exists yet for the 32B "dev" checkpoint, so Klein's value is assumed to carry over since all FLUX.2 variants share the same VAE.

Two things drive the calibration:

- **Few-step models need full-resolution steps more than they need the speedup.** At 8 steps a transition at sigma 0.90 leaves 5 steps at full resolution; the `wan21` threshold of 0.69 leaves 3, and the result goes soft.
- **The schedule shift moves where a sigma lands.** Krea 2 runs `mu=1.15`, i.e. an effective shift of `exp(1.15) = 3.158` (some ComfyUI workflows raise it to 1.5 -> 4.48), Z-Image Turbo runs a shift of 3.0 (ComfyUI) to 9.0 (Forge Neo preset), and FLUX.2 runs `mu=2.02` (effective 7.54) — all three hold sigma near 1.0 well into the run, so a threshold that reads as "early" actually lands mid-schedule or later.

`delta` stays the speed/quality slider on top of the preset — smaller transitions earlier (better quality, less speedup) — and it keeps working with adaptive delta on. Halving `delta` is a reasonable "more quality" step:

| `delta` | `flux` @1024 | `flux` @1536 | `krea-2` @ref |
|---|---|---|---|
| 0.005 | 10/40 coarse | 23/40 | 8/40 |
| **0.01** (default) | **11/40** | **25/40** | **11/40** |
| 0.02 | 13/40 | 27/40 | 14/40 |
| 0.05 | 16/40 | 30/40 | 18/40 |

(`flux` columns measured on Beta(0.6,0.6); `krea-2` at its own reference. The same `delta` buys many more coarse steps at 1536 than at 1024 — that is the point, see [the sigma boundary](#the-quality-boundary-is-a-sigma).)

Presets are for **flow-matching** models. On eps-prediction checkpoints (SD1.x / SDXL) the delta-optimal threshold is compared against a sigma scale that runs to ~14.6 rather than 1.0, so every preset transitions far later than its own criterion intends; see `docs/sdxl-analysis.md`.

### Adaptive delta (on by default)

The sigma\* figures above are for a 1024x1024 generation, and `delta_optimal` is resolution-aware: Eq. 9/10 evaluate the VAE spectrum at the coarse grid's actual Nyquist frequency in latent pixels, not as a fraction of the canvas. So the *same* preset and delta drift toward a later transition as the canvas grows — for `flux` at 1536x1536 that is 29 of 40 steps coarse instead of 25.

**Adaptive delta** pins the threshold to its reference value, `t* = max(t*_reference, t*_actual)`. It is deliberately one-directional — it only ever moves the transition *earlier*, never later — which makes it a no-op at or below 1024x1024, so small canvases don't silently become more aggressive (at 512x512 a `0.5` first scale is a 256px coarse pass, marginal for these models already).

For `flux` and `flux2` that is the whole mechanism, and it is enough: a sigma is invariant across schedules, so the step split it produces adapts on its own. The only other bound is `MAX_COARSE_FRACTION` (80%), a backstop against degeneration rather than a budget — at 40 steps it guarantees 8 full-resolution steps, and measured output only breaks below about 6. It never fires on a configuration anyone runs.

The five presets still on a spectrum-derived threshold (`krea-2`, `krea-2-raw`, `z-image`, `wan21`, `anima`) additionally hold the coarse **share**, because an uncalibrated sigma lands too late once the schedule is compressed. ComfyUI's `ModelSamplingFlux` and SwarmUI's **Sigma Shift** derive mu from `width*height`, and a FLUX.2-class schedule at 1536x1536 holds sigma above 0.93 for 30 of 40 steps; unheld, such a run gives 33/40 coarse, and at 3072x3072 the transition never fires at all and the output stays half-resolution. Those presets declare `ref_shift`, the model's effective schedule multiplier at 1024x1024, and inverting that schedule gives the share of steps sitting above the threshold there. A preset with a measured sigma carries no `ref_shift` — the cap would override it, demanding 5 of 40 steps for `flux` where the measured-correct answer is 11.

One caveat on that cap, worth knowing before you read a `[SPEED] adaptive delta: holding the ...` line as a bug: `reference_coarse_fraction` inverts a schedule laid out **uniformly in flow time**, while Beta(0.6,0.6) — the default scheduler for several of these models — clusters steps at both ends of the range and so puts more of them above a high sigma. The cap therefore trims a little even at the reference resolution, and the gap widens with the step budget: one step for `krea-2` at 20, one to two for `krea-2-raw` at 40-50, three for `anima` at its 32-step default (9/32 uncapped against 6/32 capped). Shipping conservative is the intended behaviour for a threshold nobody has A/B'd yet, but if your own calibration clears the uncapped crossing, the fix is to promote that preset to measured and drop its `ref_shift` — not to loosen the cap for every preset at once.

Turn it off to reproduce pre-adaptive behaviour exactly. `delta` keeps its meaning either way — with adaptive on it is the value **at the 1024x1024 reference**, i.e. exactly the delta the preset table above is documented against, applied at any resolution. The console reports what it did:

```
[SPEED] adaptive delta: latent 192x192 vs reference 128x128; sigma* 0.9518->0.9680 (delta 0.01 -> 0.00567 equivalent at this resolution)
[SPEED] transitions: step 25: 0.5->1 (sigma=0.9656)
[SPEED] schedule split: 25/40 steps coarse (62%), 15 at full resolution
```

### manual mode

Set `mode=manual` and supply comma-separated sigma thresholds in `manual_sigmas` (one per transition between adjacent scales). Sigma decreases as denoising progresses, so the first threshold should be larger than the last (e.g. `0.95,0.85` for three scales).

Because the quality boundary is a sigma, manual mode is the most direct expression of it: `manual_sigmas=0.968` reproduces what `flux`/`flux2` do at any resolution, on any schedule, with no preset or adaptive logic involved. Useful for calibrating a model that has no preset yet — sweep the threshold, find where your hardest detail (small text is a good probe) starts to fail, and back off from there.

### neo_shift mode (Forge Neo only)

The measured sigma anchor (`0.9680`) was calibrated where the effective schedule shift grows with resolution — ComfyUI's `ModelSamplingFlux` and SwarmUI's Sigma Shift both derive `mu` from `width*height`. Forge Neo (`sd-webui-forge-classic`) does not: Flux and FLUX.2 run at a flat `mu=1.15` there regardless of canvas size, and neither model exposes a user-facing Shift slider to change that (unlike Krea 2 / Z-Image, which do). On that flatter schedule the same sigma threshold crosses much earlier in step-count terms, so `delta_optimal`'s shared anchor is safely inside the quality boundary on Forge Neo but leaves real speedup on the table.

A live binary search on Forge Neo (Euler/Beta, seed 42, 1024x1024, `scales=0.5,1.0`) found the true quality floor for `flux` there is sigma ≈ 0.9249, not 0.968 — and reproduced that same floor at both 20 and 40 total steps once the schedule (shift + spacing) was held fixed, confirming sigma is still the invariant *within one schedule family*, just not portable across a schedule-shift change:

| sigma at transition | result |
|---|---|
| ≥ 0.9249 | perfect |
| 0.85 – 0.91 | correct text, fonts distorted |
| 0.78 – 0.83 | readable, one wrong letter |
| ≤ 0.67 | unreadable |

`neo_shift` is `delta_optimal` plus one more number, **Sigma divisor**: every resolved `t_star` (after adaptive-delta pinning, if on) is divided by it before the step lookup. `1.0` is an exact no-op — mathematically identical to `delta_optimal`, not just approximately — and the default, `1.03`, lowers the reference-case threshold to `0.9680 / 1.03 = 0.9398`, above the measured 0.9249 floor with a margin favoring quality over speed. It applies uniformly to every preset and every transition; unlike the calibrated sigma\* figures, this is a per-*backend* correction the user supplies, not a per-model constant, so there's no reason to restrict it to `flux`/`flux2` — dial it back to `1.0` for any preset/scene where it doesn't hold.

This only exists in the Forge Neo script (`scripts/speed_forge.py`); the shared `speed_core.py` gained one backward-compatible parameter (`sigma_divisor`, default `1.0`) to support it, and neither the ComfyUI node nor the SwarmUI extension expose it or are affected.

## Credits & license

This implementation is based on and derived from the official SPEED repository by Howard Xiao et al.:

- **Official code:** https://github.com/howardhx/speed (BSD 3-Clause)
- **Paper:** Xiao, H., Chao, B., Yariv, L., & Wetzstein, G. (2026). *Spectral Progressive Diffusion for Efficient Image and Video Generation.*

Please see the original project page and repository for full authorship, details, and license information.

## BibTeX

```
@article{xiao2026spectral,
  author    = {Xiao, Howard and Chao, Brian and Yariv, Lior and Wetzstein, Gordon},
  title     = {Spectral Progressive Diffusion for Efficient Image and Video Generation},
  year      = {2026},
}
```
