using FreneticUtilities.FreneticExtensions;
using Newtonsoft.Json.Linq;
using SwarmUI.Builtin_ComfyUIBackend;
using SwarmUI.Core;
using SwarmUI.Text2Image;
using SwarmUI.Utils;

namespace Ruwwww.SpeedSampler;

/// <summary>SwarmUI extension for SPEED (Spectral Progressive Diffusion).
/// Based on https://github.com/howardhx/speed — runs early denoising steps at a reduced
/// latent resolution, then spectrally expands to full resolution, for faster sampling.
/// The comfy-side math is the SamplerSPEED node from this same repo (auto-installable).</summary>
public class SPEEDSamplerExtension : Extension
{
    public const string FeatureId = "speed";

    /// <summary>Repo the backend node pack is installed from. This must be the same repo this extension ships in: SwarmUI sends node inputs by name and ComfyUI silently drops any the installed node does not declare, so a node pack from a different (or older) repo turns every parameter added here into a no-op with no error anywhere. SwarmUI clones this into DLNodes/&lt;repo name&gt;, skips the clone if that folder already exists, and then git-pulls it on every startup - so pointing this at the wrong repo is sticky and self-healing in the wrong direction.</summary>
    public const string RepoUrl = "https://github.com/aoleg/ComfyUI-SPEED";

    /// <summary>Node inputs this extension sends that older node packs do not have. Used only to warn about a version mismatch.</summary>
    public static string[] RequiredNodeInputs = ["adaptive_delta", "log"];

    /// <summary>Presets offered here, as "id///Display Name". The ids must match the node's own preset table (_PRESETS in speed_core.py) - kept in one place so the mismatch check below has something to compare against.</summary>
    public static string[] ModelPresets =
    [
        "flux///Flux",
        "wan21///Wan 2.1",
        "krea-2///Krea 2 (Turbo / Turbo LoRA)",
        "krea-2-raw///Krea 2 (raw base, CFG)",
        "z-image///Z-Image Turbo",
        "flux2///FLUX.2 (dev / Klein)",
        "anima///Anima",
        "custom///Custom (A / beta below)"
    ];

    public static T2IRegisteredParam<string> Transform, Mode, ModelPreset, Scales, ManualSigmas;

    public static T2IRegisteredParam<double> Delta, SpectrumA, SpectrumBeta;

    public static T2IRegisteredParam<bool> AdaptiveDelta;

    public static T2IRegisteredParam<long> SpectralSeed;

    public static T2IParamGroup SpeedGroup;

    /// <summary>Scheduler names that comfy's stock BasicScheduler node accepts (Swarm-custom schedulers like 'align_your_steps' or 'flux2' are computed inside SwarmKSampler and can't be rebuilt from stock nodes).</summary>
    public static HashSet<string> BasicSchedulers = ["normal", "karras", "exponential", "sgm_uniform", "simple", "ddim_uniform", "beta", "linear_quadratic", "kl_optimal"];

    /// <summary>Process-wide key for the <see cref="ClaimInit"/> guard.</summary>
    public const string InitClaimKey = "Ruwwww.SpeedSampler.Initialized";

    /// <summary>Claims the one-time half of <see cref="OnInit"/>, returning false if it is already done.
    ///
    /// The "install" button in the SwarmUI UI clones this same repo into
    /// BuiltinExtensions/ComfyUIBackend/DLNodes/ComfyUI-SPEED/ to supply the comfy-side node - that's required (the node
    /// pack must match this extension's repo). But SwarmUI.csproj's default compile glob only excludes Extensions/**, not
    /// BuiltinExtensions/**/DLNodes/**, so that clone's copy of this .cs file also gets compiled into the core SwarmUI
    /// assembly, producing a second SPEEDSamplerExtension instance whose OnInit() runs alongside the real one loaded from
    /// Extensions/ComfyUI-SPEED/.
    ///
    /// The claim cannot be a plain 'private static bool': those two instances are distinct types in distinct assemblies
    /// and so have distinct statics, leaving the second copy's flag false and the guard doing nothing. Verified against
    /// SwarmUI 2db744ed, where a static guard still let startup throw twice - once at the first T2IParamTypes.Register
    /// (T2IParamTypes.cs:286, key 'speedtransform'), which aborts the second OnInit before WorkflowGenerator.AddStep, and
    /// once more in WebServer.GatherExtensionPageAdditions (WebServer.cs:330) over the duplicated ScriptFiles entries,
    /// which were added before that first throw. Both log as non-fatal, so SPEED reads as half-broken rather than as a
    /// clean failure. AppDomain data is one slot for the whole process and is visible to both copies.
    ///
    /// Extension init runs on a single thread (ExtensionsManager.RunOnAllExtensions), so check-then-set needs no lock.</summary>
    private static bool ClaimInit()
    {
        if (AppDomain.CurrentDomain.GetData(InitClaimKey) is not null)
        {
            return false;
        }
        AppDomain.CurrentDomain.SetData(InitClaimKey, InitClaimKey);
        return true;
    }

    public override void PopulateMetadata()
    {
        base.PopulateMetadata();
        Description = "Runs early denoising steps at a reduced latent resolution and spectrally expands to full resolution for faster sampling of flow-matching models.";
        Tags = ["parameters", "nodes"];
        License = "MIT";
    }

    public override void OnInit()
    {
        InstallableFeatures.RegisterInstallableFeature(new("SPEED Sampler", FeatureId, RepoUrl, "aoleg", $"This will install the ComfyUI-SPEED node pack from {RepoUrl} (MIT License, derived from ruwwww's ComfyUI-SPEED and from the official SPEED code by Howard Xiao et al, BSD-3-Clause).\nThis must be the same repo this extension came from - the model presets and adaptive-delta options it sends need the matching node version.\nDo you wish to install?"));
        // Indexer assignment is naturally idempotent, so it stays outside the guard and the node stays
        // mapped to the feature whichever copy of this class wins the claim.
        ComfyUIBackendExtension.NodeToFeatureMap["SamplerSPEED"] = FeatureId;
        if (!ClaimInit())
        {
            Logs.Debug("[SPEED] SPEEDSamplerExtension.OnInit() ran again (duplicate copy compiled from BuiltinExtensions/ComfyUIBackend/DLNodes/ComfyUI-SPEED/); skipping duplicate param/script registration.");
            return;
        }
        ScriptFiles.Add("assets/speed_install.js");
        ScriptFiles.Add("assets/speed_ui.js");
        ComfyUIBackendExtension.RawObjectInfoParsers.Add(WarnOnNodeVersionMismatch);
        SpeedGroup = new("SPEED", Toggles: true, Open: false, IsAdvanced: true,
            Description: "SPEED (Spectral Progressive Diffusion): runs early denoising steps at a reduced latent resolution, then spectrally expands to full resolution, reducing computation while preserving visual quality.\nDesigned for flow-matching models (Flux, Wan, Krea 2, Z-Image). The main 'Sampler' parameter is used as the base solver.\nNot applied when a mask is in use, and not applied to the Refiner stage.");
        Transform = T2IParamTypes.Register<string>(new("[SPEED] Transform", "[SPEED]\nSpectral basis used at each resolution transition.\nDCT (default) supports any scale ratio. DWT requires consecutive scales to differ by exactly 2x, and needs the optional 'PyWavelets' pip package on the backend. FFT accepts any ratio.",
            "dct", Group: SpeedGroup, FeatureFlag: FeatureId, OrderPriority: 1,
            GetValues: (_) => ["dct///DCT (any ratio)", "dwt///DWT (2x only, needs PyWavelets)", "fft///FFT (any ratio)"]
            ));
        Mode = T2IParamTypes.Register<string>(new("[SPEED] Mode", "[SPEED]\n'delta_optimal' computes transition timing automatically from '[SPEED] Scales', '[SPEED] Delta', and the selected power-spectrum preset (Eq. 9/10 of the paper).\n'manual' uses the explicit sigma thresholds from '[SPEED] Manual Sigmas'.",
            "delta_optimal", Group: SpeedGroup, FeatureFlag: FeatureId, OrderPriority: 2,
            GetValues: (_) => ["delta_optimal///Delta-Optimal (automatic)", "manual///Manual sigma thresholds"]
            ));
        ModelPreset = T2IParamTypes.Register<string>(new("[SPEED] Model Preset", "[SPEED]\nPower-spectrum preset for delta-optimal mode.\n'flux' and 'flux2' are calibrated from measured A/B output and pin a transition sigma of 0.968 - the boundary below which fine detail (small text is the harshest probe) starts to fail. Note 'flux' therefore no longer ships the paper's published fit; select 'custom' with A 203.615097 / beta 1.915461 to restore it.\n'wan21' is the upstream published fit.\n'krea-2' targets few-step Krea 2 - the Turbo checkpoint at ~8 steps, or the base model with the Turbo LoRA at ~20 steps.\n'krea-2-raw' targets the undistilled Krea 2 base model at 40-50 steps with CFG, and transitions later for more speedup.\n'z-image' targets Z-Image Turbo.\n'anima' targets Anima (Cosmos-Predict2 DiT on the Qwen-Image/Wan 2.1 VAE) at its 32-step default. Those four are spectrum-derived and not yet A/B tested.\n'custom' uses the '[SPEED] Spectrum A' / '[SPEED] Spectrum Beta' inputs, which only show once 'custom' is selected.\nsigma* is quoted at 1024x1024. Watch the backend console's '[SPEED] transitions:' line to see where a run actually landed.",
            "flux", Group: SpeedGroup, FeatureFlag: FeatureId, OrderPriority: 3,
            GetValues: (_) => [.. ModelPresets]
            ));
        Scales = T2IParamTypes.Register<string>(new("[SPEED] Scales", "[SPEED]\nComma-separated resolution fractions ending at 1.0.\nFor example '0.5,1.0' runs early steps at half resolution, or '0.25,0.5,1.0' adds a quarter-resolution stage.",
            "0.5,1.0", Group: SpeedGroup, FeatureFlag: FeatureId, OrderPriority: 4,
            Examples: ["0.5,1.0", "0.25,0.5,1.0", "0.75,1.0"]
            ));
        AdaptiveDelta = T2IParamTypes.Register<bool>(new("[SPEED] Adaptive Delta", "[SPEED]\nKeeps the coarse/detail step split at the split the preset was calibrated for (1024x1024) instead of letting the coarse pass grow with the canvas.\nEq. 10 evaluates the latent spectrum at the coarse grid's own Nyquist frequency, so without this the same Delta spends progressively more of the schedule at low resolution as you generate larger - with 'flux' at 40 steps that is 16 of 40 coarse steps at 1024x1024 but 22 of 40 at 2048x2048, starving the detail pass exactly where you need it most.\nThis only ever moves the transition earlier, never later, so it is a no-op at or below 1024x1024 and needs no retuning per resolution.\nApplies to delta-optimal mode only ('manual' sigma thresholds are already resolution-independent).\nTurn this off to reproduce pre-adaptive behaviour exactly.",
            "true", Group: SpeedGroup, FeatureFlag: FeatureId, OrderPriority: 4.5
            ));
        Delta = T2IParamTypes.Register<double>(new("[SPEED] Delta", "[SPEED]\nNoise-dominated tolerance for delta-optimal mode.\nSmaller values delay the transitions to full resolution (higher quality, less speedup).\nWith '[SPEED] Adaptive Delta' on, this is the value at the 1024x1024 reference - exactly the delta the preset table is documented against, applied at any resolution.\nIf a transition never fits in the schedule, the backend logs a warning and the output stays coarse - raise this or add steps.",
            "0.01", Min: 0.0001, Max: 0.5, Step: 0.001, Group: SpeedGroup, FeatureFlag: FeatureId, OrderPriority: 5,
            Examples: ["0.005", "0.01", "0.05"]
            ));
        ManualSigmas = T2IParamTypes.Register<string>(new("[SPEED] Manual Sigmas", "[SPEED]\nComma-separated sigma thresholds, one per transition (length must be one less than the scales list), strictly decreasing.\nOnly used in 'manual' mode. Example for scales '0.25,0.5,1.0': '0.95,0.85'.",
            "0.85", Group: SpeedGroup, FeatureFlag: FeatureId, OrderPriority: 6, DependNonDefault: Mode.Type.ID,
            Examples: ["0.85", "0.95,0.85"]
            ));
        // No DependNonDefault here: that mechanism only supports "other param differs from
        // its own default" (here, "!= flux"), which would show these on every non-flux
        // preset. Visibility for these two is instead handled entirely by
        // assets/speed_ui.js, gated on ModelPreset == "custom" specifically.
        SpectrumA = T2IParamTypes.Register<double>(new("[SPEED] Spectrum A", "[SPEED]\nPower-law amplitude 'A' of the latent power spectrum.\nOnly used when '[SPEED] Model Preset' is 'custom'.",
            "203.615097", Min: 0, Max: 1000000, Step: 0.001, Group: SpeedGroup, FeatureFlag: FeatureId, OrderPriority: 7
            ));
        SpectrumBeta = T2IParamTypes.Register<double>(new("[SPEED] Spectrum Beta", "[SPEED]\nPower-law decay exponent 'beta' of the latent power spectrum.\nOnly used when '[SPEED] Model Preset' is 'custom'.",
            "1.915461", Min: 0, Max: 10, Step: 0.001, Group: SpeedGroup, FeatureFlag: FeatureId, OrderPriority: 8
            ));
        SpectralSeed = T2IParamTypes.Register<long>(new("[SPEED] Spectral Noise Seed", "[SPEED]\nSeed for the spectral-noise padding applied at each transition.\n'-1' follows the generation seed.",
            "-1", Min: -1, Max: int.MaxValue, Group: SpeedGroup, FeatureFlag: FeatureId, OrderPriority: 9
            ));
        // Runs just after the main sampler step (-5), before the refiner (-4): rewrites the main
        // sampler node in place so all downstream connections to it remain valid.
        WorkflowGenerator.AddStep(ApplySpeed, -4.5);
    }

    /// <summary>Warns when the installed node pack is older than this extension.
    /// ComfyUI builds a node's arguments from its own INPUT_TYPES and ignores anything else in the prompt, so sending a parameter
    /// a stale node does not declare is silently dropped - the control appears in the UI, changes nothing, and reports no error.
    /// That is expensive to debug from the symptom, so surface it at the one moment we can actually see the node's schema.</summary>
    public static void WarnOnNodeVersionMismatch(JObject rawObjectInfo)
    {
        if (rawObjectInfo["SamplerSPEED"] is not JObject node)
        {
            return; // Not installed at all; the feature flag already covers that case.
        }
        HashSet<string> declared = [];
        foreach (string section in new[] { "required", "optional" })
        {
            if (node["input"]?[section] is JObject inputs)
            {
                foreach (JProperty prop in inputs.Properties())
                {
                    declared.Add(prop.Name);
                }
            }
        }
        string[] missingInputs = [.. RequiredNodeInputs.Where(i => !declared.Contains(i))];
        if (missingInputs.Any())
        {
            Logs.Warning($"[SPEED] The installed ComfyUI-SPEED node pack is older than this extension: it does not accept {string.Join(", ", missingInputs)}. "
                + "ComfyUI ignores inputs a node does not declare, so those options will do nothing (with no error). "
                + $"Update the node pack - its git remote must be {RepoUrl}.");
        }
        // The preset list lives in the node, so a stale pack also rejects any preset added since. That one fails loudly at
        // generation time rather than silently, but naming it here is far more actionable than 'value not in list'.
        if (ComfyUIBackendExtension.TryGetRequiredInputs(rawObjectInfo, "SamplerSPEED", "model_preset", out JToken presets))
        {
            HashSet<string> available = [.. presets.Select(p => $"{p}")];
            string[] missingPresets = [.. ModelPresets.Select(v => v.Before("///")).Where(v => !available.Contains(v))];
            if (missingPresets.Any())
            {
                Logs.Warning($"[SPEED] The installed node pack does not offer these presets: {string.Join(", ", missingPresets)}. "
                    + "Selecting one will fail the generation. Update the node pack.");
            }
        }
    }

    public static void ApplySpeed(WorkflowGenerator g)
    {
        // Any one group-param being present indicates the SPEED group toggle is enabled.
        if (!g.UserInput.TryGet(Transform, out string transform))
        {
            return;
        }
        if (!g.Features.Contains(FeatureId))
        {
            throw new SwarmUserErrorException("SPEED parameters specified, but the ComfyUI-SPEED node pack isn't installed on the backend.");
        }
        if (g.FinalMask is not null)
        {
            Logs.Warning("[SPEED] a mask is in use; SPEED skipped (masks are tied to the full latent resolution).");
            return;
        }
        // "10" is the fixed ID the core generator gives the main (base stage) sampler node.
        if (g.Workflow["10"] is not JObject samplerNode || samplerNode["inputs"] is not JObject samplerInputs)
        {
            Logs.Warning("[SPEED] no main sampler node found in the workflow; SPEED skipped.");
            return;
        }
        string createSpeedSampler(string baseSampler)
        {
            long spectralSeed = g.UserInput.Get(SpectralSeed, -1);
            if (spectralSeed < 0)
            {
                spectralSeed = g.UserInput.Get(T2IParamTypes.Seed, 0);
            }
            return g.CreateNode("SamplerSPEED", new JObject()
            {
                ["base_sampler"] = baseSampler,
                ["transform"] = transform,
                ["mode"] = g.UserInput.Get(Mode, "delta_optimal"),
                ["model_preset"] = g.UserInput.Get(ModelPreset, "flux"),
                ["scales"] = g.UserInput.Get(Scales, "0.5,1.0"),
                ["adaptive_delta"] = g.UserInput.Get(AdaptiveDelta, true),
                ["delta"] = g.UserInput.Get(Delta, 0.01),
                ["manual_sigmas"] = g.UserInput.Get(ManualSigmas, "0.85"),
                ["spectrum_A"] = g.UserInput.Get(SpectrumA, 203.615097),
                ["spectrum_beta"] = g.UserInput.Get(SpectrumBeta, 1.915461),
                ["seed"] = spectralSeed % int.MaxValue,
                ["log"] = true
            });
        }
        string classType = $"{samplerNode["class_type"]}";
        if (classType == "SamplerCustomAdvanced")
        {
            // Core already emitted a custom-sampling graph (eg dual-guider/ip2p models): just swap the SAMPLER input.
            string baseSampler = "euler";
            if (samplerInputs["sampler"] is JArray samplerRef && g.Workflow[$"{samplerRef[0]}"] is JObject selectNode
                && $"{selectNode["class_type"]}" == "KSamplerSelect")
            {
                baseSampler = $"{selectNode["inputs"]["sampler_name"]}";
            }
            samplerInputs["sampler"] = new JArray() { createSpeedSampler(baseSampler), 0 };
            return;
        }
        if (classType != "SwarmKSampler" && classType != "KSamplerAdvanced")
        {
            Logs.Warning($"[SPEED] main sampler node is a '{classType}', which SPEED doesn't know how to rebuild; SPEED skipped.");
            return;
        }
        // SPEED must be applied as a SAMPLER object, so rebuild the KSampler as a
        // SamplerCustomAdvanced graph (mirroring core's own emitAsCustomAdvanced logic).
        JToken model = samplerInputs["model"], pos = samplerInputs["positive"], neg = samplerInputs["negative"], latent = samplerInputs["latent_image"];
        long seed = (long)samplerInputs["noise_seed"];
        int steps = (int)samplerInputs["steps"];
        double cfg = (double)samplerInputs["cfg"];
        string samplerName = $"{samplerInputs["sampler_name"]}";
        string scheduler = $"{samplerInputs["scheduler"]}".ToLowerFast();
        int startStep = (int)samplerInputs["start_at_step"];
        int endStep = (int)samplerInputs["end_at_step"];
        bool addNoise = $"{samplerInputs["add_noise"]}" == "enable";
        double sigmin = samplerInputs.TryGetValue("sigma_min", out JToken sigminTok) ? (double)sigminTok : g.UserInput.Get(T2IParamTypes.SamplerSigmaMin, -1);
        double sigmax = samplerInputs.TryGetValue("sigma_max", out JToken sigmaxTok) ? (double)sigmaxTok : g.UserInput.Get(T2IParamTypes.SamplerSigmaMax, -1);
        double rho = samplerInputs.TryGetValue("rho", out JToken rhoTok) ? (double)rhoTok : g.UserInput.Get(T2IParamTypes.SamplerRho, 7);
        if (samplerInputs.TryGetValue("var_seed_strength", out JToken varStrength) && (double)varStrength > 0)
        {
            Logs.Warning("[SPEED] Variation Seed is not supported together with SPEED and will be ignored.");
        }
        string noiseNode = addNoise
            ? g.CreateNode("RandomNoise", new JObject()
            {
                ["noise_seed"] = seed
            })
            : g.CreateNode("DisableNoise", new JObject());
        string guiderNode = g.CreateNode("CFGGuider", new JObject()
        {
            ["model"] = model,
            ["positive"] = pos,
            ["negative"] = neg,
            ["cfg"] = cfg
        });
        JArray sigmas;
        if (scheduler == "turbo")
        {
            string turboNode = g.CreateNode("SDTurboScheduler", new JObject()
            {
                ["model"] = model,
                ["steps"] = steps,
                ["denoise"] = 1
            });
            sigmas = [turboNode, 0];
        }
        else if (scheduler == "karras")
        {
            string karrasNode = g.CreateNode("KarrasScheduler", new JObject()
            {
                ["steps"] = steps,
                ["sigma_max"] = sigmax <= 0 ? 14.614642 : sigmax,
                ["sigma_min"] = sigmin <= 0 ? 0.0291675 : sigmin,
                ["rho"] = rho
            });
            sigmas = [karrasNode, 0];
        }
        else
        {
            if (!BasicSchedulers.Contains(scheduler))
            {
                Logs.Warning($"[SPEED] scheduler '{scheduler}' is Swarm-custom and can't be rebuilt with stock comfy nodes; using 'simple' instead.");
                scheduler = "simple";
            }
            string basicNode = g.CreateNode("BasicScheduler", new JObject()
            {
                ["model"] = model,
                ["steps"] = steps,
                ["scheduler"] = scheduler,
                ["denoise"] = 1
            });
            sigmas = [basicNode, 0];
        }
        if (startStep > 0)
        {
            string afterStart = g.CreateNode("SplitSigmas", new JObject()
            {
                ["sigmas"] = sigmas,
                ["step"] = startStep
            });
            sigmas = [afterStart, 1];
        }
        if (endStep < steps)
        {
            string beforeEnd = g.CreateNode("SplitSigmas", new JObject()
            {
                ["sigmas"] = sigmas,
                ["step"] = endStep
            });
            sigmas = [beforeEnd, 0];
        }
        samplerNode["class_type"] = "SamplerCustomAdvanced";
        samplerNode["inputs"] = new JObject()
        {
            ["noise"] = new JArray() { noiseNode, 0 },
            ["guider"] = new JArray() { guiderNode, 0 },
            ["sampler"] = new JArray() { createSpeedSampler(samplerName), 0 },
            ["sigmas"] = sigmas,
            ["latent_image"] = latent
        };
        Logs.Debug("[SPEED] main sampler rebuilt as SamplerCustomAdvanced + SamplerSPEED.");
    }
}
