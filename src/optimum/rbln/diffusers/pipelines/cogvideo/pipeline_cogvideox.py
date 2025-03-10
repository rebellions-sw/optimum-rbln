from diffusers import CogVideoXPipeline

from ...modeling_diffusers import RBLNDiffusionMixin


class RBLNCogVideoXPipeline(RBLNDiffusionMixin, CogVideoXPipeline):
    original_class = CogVideoXPipeline
    # _submodules = ["text_encoder", "transformer", "vae"]
    _submodules = ["transformer"]
    # _submodules = ["vae"]
    # _submodules = ["text_encoder"]
