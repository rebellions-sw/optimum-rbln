from diffusers import CogVideoXPipeline

from ...modeling_diffusers import RBLNDiffusionMixin


class RBLNCogVideoXPipeline(RBLNDiffusionMixin, CogVideoXPipeline):
    # _submodules = ["text_encoder", "transformer", "vae"]
    _submodules = ["transformer"]