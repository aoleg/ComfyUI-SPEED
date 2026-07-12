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

    public static T2IRegisteredParam<string> Transform, Mode, ModelPreset, Scales, ManualSigmas;

    public static T2IRegisteredParam<double> Delta, SpectrumA, SpectrumBeta;

    public static T2IRegisteredParam<long> SpectralSeed;

    public static T2IParamGroup SpeedGroup;

    /// <summary>Scheduler names that comfy's stock BasicScheduler node accepts (Swarm-custom schedulers like 'align_your_steps' or 'flux2' are computed inside SwarmKSampler and can't be rebuilt from stock nodes).</summary>
    public static HashSet<string> BasicSchedulers = ["normal", "karras", "exponential", "sgm_uniform", "simple", "ddim_uniform", "beta", "linear_quadratic", "kl_optimal"];

    public override void OnInit()
    {
        InstallableFeatures.RegisterInstallableFeature(new("SPEED Sampler", FeatureId, "https://github.com/ruwwww/ComfyUI-SPEED", "ruwwww", "This will install the ComfyUI-SPEED node pack by ruwwww (MIT License, derived from the official SPEED code by Howard Xiao et al, BSD-3-Clause).\nDo you wish to install?"));
        ScriptFiles.Add("assets/speed_install.js");
        ComfyUIBackendExtension.NodeToFeatureMap["SamplerSPEED"] = FeatureId;
        SpeedGroup = new("SPEED", Toggles: true, Open: false, IsAdvanced: true,
            Description: "SPEED (Spectral Progressive Diffusion): runs early denoising steps at a reduced latent resolution, then spectrally expands to full resolution, reducing computation while preserving visual quality.\nDesigned for flow-matching models (Flux, Wan). The main 'Sampler' parameter is used as the base solver.\nNot applied when a mask is in use, and not applied to the Refiner stage.");
        Transform = T2IParamTypes.Register<string>(new("[SPEED] Transform", "[SPEED]\nSpectral basis used at each resolution transition.\nDCT (default) supports any scale ratio. DWT requires consecutive scales to differ by exactly 2x, and needs the optional 'PyWavelets' pip package on the backend. FFT accepts any ratio.",
            "dct", Group: SpeedGroup, FeatureFlag: FeatureId, OrderPriority: 1,
            GetValues: (_) => ["dct///DCT (any ratio)", "dwt///DWT (2x only, needs PyWavelets)", "fft///FFT (any ratio)"]
            ));
        Mode = T2IParamTypes.Register<string>(new("[SPEED] Mode", "[SPEED]\n'delta_optimal' computes transition timing automatically from '[SPEED] Scales', '[SPEED] Delta', and the selected power-spectrum preset (Eq. 9/10 of the paper).\n'manual' uses the explicit sigma thresholds from '[SPEED] Manual Sigmas'.",
            "delta_optimal", Group: SpeedGroup, FeatureFlag: FeatureId, OrderPriority: 2,
            GetValues: (_) => ["delta_optimal///Delta-Optimal (automatic)", "manual///Manual sigma thresholds"]
            ));
        ModelPreset = T2IParamTypes.Register<string>(new("[SPEED] Model Preset", "[SPEED]\nPower-spectrum preset for delta-optimal mode.\n'flux' and 'wan21' use measured (A, beta) fits. 'custom' uses the '[SPEED] Spectrum A' / '[SPEED] Spectrum Beta' inputs.",
            "flux", Group: SpeedGroup, FeatureFlag: FeatureId, OrderPriority: 3,
            GetValues: (_) => ["flux", "wan21", "custom"]
            ));
        Scales = T2IParamTypes.Register<string>(new("[SPEED] Scales", "[SPEED]\nComma-separated resolution fractions ending at 1.0.\nFor example '0.5,1.0' runs early steps at half resolution, or '0.25,0.5,1.0' adds a quarter-resolution stage.",
            "0.5,1.0", Group: SpeedGroup, FeatureFlag: FeatureId, OrderPriority: 4,
            Examples: ["0.5,1.0", "0.25,0.5,1.0", "0.75,1.0"]
            ));
        Delta = T2IParamTypes.Register<double>(new("[SPEED] Delta", "[SPEED]\nNoise-dominated tolerance for delta-optimal mode.\nSmaller values delay the transitions to full resolution (higher quality, less speedup).\nIf a transition never fits in the schedule, the backend logs a warning and the output stays coarse - raise this or add steps.",
            "0.01", Min: 0.0001, Max: 0.5, Step: 0.001, Group: SpeedGroup, FeatureFlag: FeatureId, OrderPriority: 5,
            Examples: ["0.005", "0.01", "0.05"]
            ));
        ManualSigmas = T2IParamTypes.Register<string>(new("[SPEED] Manual Sigmas", "[SPEED]\nComma-separated sigma thresholds, one per transition (length must be one less than the scales list), strictly decreasing.\nOnly used in 'manual' mode. Example for scales '0.25,0.5,1.0': '0.95,0.85'.",
            "0.85", Group: SpeedGroup, FeatureFlag: FeatureId, OrderPriority: 6, DependNonDefault: Mode.Type.ID,
            Examples: ["0.85", "0.95,0.85"]
            ));
        SpectrumA = T2IParamTypes.Register<double>(new("[SPEED] Spectrum A", "[SPEED]\nPower-law amplitude 'A' of the latent power spectrum.\nOnly used when '[SPEED] Model Preset' is 'custom'.",
            "203.615097", Min: 0, Max: 1000000, Step: 0.001, Group: SpeedGroup, FeatureFlag: FeatureId, OrderPriority: 7, DependNonDefault: ModelPreset.Type.ID
            ));
        SpectrumBeta = T2IParamTypes.Register<double>(new("[SPEED] Spectrum Beta", "[SPEED]\nPower-law decay exponent 'beta' of the latent power spectrum.\nOnly used when '[SPEED] Model Preset' is 'custom'.",
            "1.915461", Min: 0, Max: 10, Step: 0.001, Group: SpeedGroup, FeatureFlag: FeatureId, OrderPriority: 8, DependNonDefault: ModelPreset.Type.ID
            ));
        SpectralSeed = T2IParamTypes.Register<long>(new("[SPEED] Spectral Noise Seed", "[SPEED]\nSeed for the spectral-noise padding applied at each transition.\n'-1' follows the generation seed.",
            "-1", Min: -1, Max: int.MaxValue, Group: SpeedGroup, FeatureFlag: FeatureId, OrderPriority: 9
            ));
        // Runs just after the main sampler step (-5), before the refiner (-4): rewrites the main
        // sampler node in place so all downstream connections to it remain valid.
        WorkflowGenerator.AddStep(ApplySpeed, -4.5);
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
                ["delta"] = g.UserInput.Get(Delta, 0.01),
                ["manual_sigmas"] = g.UserInput.Get(ManualSigmas, "0.85"),
                ["spectrum_A"] = g.UserInput.Get(SpectrumA, 203.615097),
                ["spectrum_beta"] = g.UserInput.Get(SpectrumBeta, 1.915461),
                ["seed"] = spectralSeed % int.MaxValue
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
