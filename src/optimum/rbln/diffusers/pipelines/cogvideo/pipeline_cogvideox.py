from diffusers import CogVideoXPipeline

from ...modeling_diffusers import RBLNDiffusionMixin


class RBLNCogVideoXPipeline(RBLNDiffusionMixin, CogVideoXPipeline):
    # NOTE: CovVideoX1.5 is supported from diffusers >= 0.32.0
    # ref: https://github.com/huggingface/diffusers/blob/v0.32.0/src/diffusers/pipelines/cogvideo/pipeline_cogvideox.py
    original_class = CogVideoXPipeline
    # _submodules = ["text_encoder", "transformer", "vae"]
    _submodules = ["transformer"]
    # _submodules = ["vae"]
    # _submodules = ["text_encoder"]
