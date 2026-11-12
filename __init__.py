# Based on https://github.com/howardhx/speed — Spectral Progressive Diffusion
from .speed_sampler import SamplerSPEED, SamplerSPEEDHandover

NODE_CLASS_MAPPINGS = {
    "SamplerSPEED": SamplerSPEED,
    "SamplerSPEEDHandover": SamplerSPEEDHandover,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SamplerSPEED": "Sampler SPEED (Spectral Progressive Diffusion)",
    "SamplerSPEEDHandover": "Sampler SPEED x0 hand-over (distilled models)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
