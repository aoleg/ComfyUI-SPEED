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
| `model_preset` | combo | `flux` | Power-spectrum preset: `flux`, `wan21`, `krea-2`, `krea-2-raw`, `z-image`, `flux2`, or `custom` (use `spectrum_A` / `spectrum_beta`). See [Model presets](#model-presets). |
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

| Preset | Latent space | Intended for | sigma* |
|---|---|---|---|
| `flux` | Flux VAE | Flux (measured fit, from the official repo) | 0.8527 |
| `wan21` | Wan 2.1 VAE | Wan 2.1 video (measured fit, from the official repo) | 0.6941 |
| `krea-2` | Wan 2.1 / Qwen-Image | Few-step Krea 2: the Turbo checkpoint at ~8 steps, or the base model + Turbo LoRA at ~20 steps | 0.9000 |
| `krea-2-raw` | Wan 2.1 / Qwen-Image | Undistilled Krea 2 base model, 40-50 steps with CFG — transitions later, more speedup | 0.8300 |
| `z-image` | Flux VAE | Z-Image Turbo, 8-9 steps | 0.9400 |
| `flux2` | Flux VAE (**unverified**) | FLUX.2 [dev] / Klein, 40-50 steps | 0.9000 † |
| `custom` | — | Your own `spectrum_A` / `spectrum_beta` | — |

† FLUX.2's sampler latent is 1/16 scale, so its sigma\* is quoted at a 64x64 reference grid; every other preset uses 128x128. Same 1024x1024 image either way.

`flux` and `wan21` are the fits published with the paper. `krea-2`, `krea-2-raw` and `z-image` are **calibrated, not measured**: `beta` is taken from the measured fit for the same VAE latent space (Krea 2 encodes with the Qwen-Image VAE, whose latent format is the Wan 2.1 one; Z-Image uses the Flux VAE), and `A` is solved so that the default `delta` places the transition where the model's step budget wants it. The derivation is in the comment block above `_PRESETS` in `speed_core.py`, and `tests/test_math.py` asserts each sigma\* above.

`flux2` is weaker than that, and is flagged accordingly. FLUX.2's VAE encodes at 8x into 32 channels and then packs 2x2 spatially into the channel axis (`AutoencoderKLFlux2.encode`, `patch_size: [2, 2]`), so the tensor the sampler — and therefore SPEED — actually sees is 128 channels at **1/16 scale**: a 1024x1024 image is a 64x64 grid, not the 128x128 every other preset here works on. Its sigma\* is calibrated at that geometry, which is why its `A` looks small next to `flux`'s despite targeting a *later* transition. Its `beta` is Flux.1's only because that's the closest available analog (same lineage), not because the latents match in shape or channel count — treat it as a starting point and refine with `custom` if you can A/B it. What *is* source-backed: FLUX.2 Klein 4B/9B both declare `shift = 2.02` in ComfyUI's `supported_models.py` (log-space, like Flux.1/Krea 2 — an effective multiplier of `exp(2.02) = 7.54`); no separate entry exists yet for the 32B "dev" checkpoint, so Klein's value is assumed to carry over since all FLUX.2 variants share the same VAE.

Two things drive the calibration:

- **Few-step models need full-resolution steps more than they need the speedup.** At 8 steps a transition at sigma 0.90 leaves 5 steps at full resolution; the `wan21` threshold of 0.69 leaves 3, and the result goes soft.
- **The schedule shift moves where a sigma lands.** Krea 2 runs `mu=1.15`, i.e. an effective shift of `exp(1.15) = 3.158` (some ComfyUI workflows raise it to 1.5 -> 4.48), Z-Image Turbo runs a shift of 3.0 (ComfyUI) to 9.0 (Forge Neo preset), and FLUX.2 runs `mu=2.02` (effective 7.54) — all three hold sigma near 1.0 well into the run, so a threshold that reads as "early" actually lands mid-schedule or later.

`delta` stays the speed/quality slider on top of the preset — smaller transitions earlier (better quality, less speedup). For `krea-2` it spans sigma\* 0.93 at `delta=0.005` to 0.80 at `delta=0.05`.

Presets are for **flow-matching** models. On eps-prediction checkpoints (SD1.x / SDXL) the delta-optimal threshold is compared against a sigma scale that runs to ~14.6 rather than 1.0, so every preset transitions far later than its own criterion intends; see `docs/sdxl-analysis.md`.

### Adaptive delta (on by default)

The sigma\* figures above are all for a 1024x1024 generation, and `delta_optimal` is resolution-aware: Eq. 9/10 evaluate the VAE spectrum at the coarse grid's actual Nyquist frequency in latent pixels, not as a fraction of the canvas. Left alone, that means the *same* preset and delta hand a progressively larger share of the schedule to the coarse pass as the canvas grows:

| Resolution | `flux` sigma\* | 40 steps, adaptive **off** | 40 steps, adaptive **on** |
|---|---|---|---|
| 1024×1024 | 0.8527 | 15 coarse / 25 detail | 15 / 25 |
| 1536×1536 | 0.7869 | 19 coarse / 21 detail | 15 / 25 |
| 2048×2048 | 0.7324 | 22 coarse / 18 detail | 15 / 25 |
| 3072×3072 | 0.6457 | 26 coarse / 14 detail | 15 / 25 |

Eq. 10 is not wrong here — a larger canvas really does keep its high-frequency band noise-dominated for longer — but the result inverts the point of the extension. The coarse pass only lays down composition; it does not need 26 of 40 steps to do that, and the steps it takes come straight out of the detail pass, at exactly the resolutions where detail matters most. **Adaptive delta** holds the split where the preset was calibrated instead, with no per-resolution retuning.

It is deliberately one-directional — it only ever moves the transition *earlier* than the unadapted path, never later — which gives two useful guarantees:

- **A no-op at or below 1024x1024.** Below the reference the unadapted threshold is already the more conservative one and is kept, so small canvases don't silently become more aggressive (at 512x512 a `0.5` first scale is a 256px coarse pass, marginal for these models already).
- **A cap on the coarse share, derived from the reference split.** Threshold pinning alone only neutralises the resolution term. If the *schedule* is also resolution-dependent — ComfyUI's `ModelSamplingFlux` and SwarmUI's **Sigma Shift** derive mu from `width*height`, reaching mu 3.23 at 2048x2048 against 1.15 at 1024x1024 — it compresses sigma toward 1.0 and pushes the crossing later whatever the threshold says. Unguarded that degenerates badly: 37 of 40 steps coarse at 2048x2048, and at 3072x3072 the transition never fires at all and the output stays half-resolution.

  A sigma threshold simply cannot express "the same split" once the schedule itself moves, so the cap is a step **fraction** instead. Each preset declares `ref_shift`, the model's own effective schedule multiplier at 1024x1024, and inverting that schedule gives the share of steps that would sit above the threshold there — 35.3% for `flux` at `delta=0.01`, i.e. 15 of 40 steps. So a 1536x1536 render with Sigma Shift on lands on **15/40 too**, matching the reference rather than settling for a flat half-and-half:

  | With Sigma Shift on | adaptive off | adaptive on |
  |---|---|---|
  | 1024×1024 | 15 / 25 | 15 / 25 |
  | 1536×1536 | 27 / 13 | **15 / 25** |
  | 2048×2048 | 37 / 3 | **15 / 25** |
  | 3072×3072 | 40 / 0 (never transitions) | **15 / 25** |

  `custom` declares no `ref_shift` — the model, and so its schedule, is unknown — and falls back to a flat 50% bound, which also stays on as a backstop for pathological `delta` values whose reference split is itself degenerate.

Turn it off to reproduce pre-adaptive behaviour exactly. `delta` keeps its meaning either way — with adaptive on it is the value **at the 1024x1024 reference**, i.e. exactly the delta the preset table above is documented against, applied at any resolution. The console reports what it did:

```
[SPEED] adaptive delta: latent 192x192 vs reference 128x128; sigma* 0.7869->0.8527 (delta 0.01 -> 0.00409 equivalent at this resolution)
[SPEED] transitions: step 15: 0.5->1 (sigma=0.8473)
[SPEED] schedule split: 15/40 steps coarse (38%), 25 at full resolution
```

### manual mode

Set `mode=manual` and supply comma-separated sigma thresholds in `manual_sigmas` (one per transition between adjacent scales). Sigma decreases as denoising progresses, so the first threshold should be larger than the last (e.g. `0.95,0.85` for three scales).

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
