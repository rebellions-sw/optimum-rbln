# Copyright 2024 Rebellions Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from diffusers import (
    DDPMScheduler,
    KandinskyV22InpaintCombinedPipeline,
    PriorTransformer,
    UnCLIPScheduler,
    UNet2DConditionModel,
    VQModel,
)
from transformers import (
    CLIPImageProcessor,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
)

from ...modeling_diffusers import RBLNDiffusionMixin
from .pipeline_kandinsky2_2_inpaint import RBLNKandinskyV22InpaintPipeline
from .pipeline_kandinsky2_2_prior import RBLNKandinskyV22PriorPipeline


class RBLNKandinskyV22InpaintCombinedPipeline(RBLNDiffusionMixin, KandinskyV22InpaintCombinedPipeline):
    original_class = KandinskyV22InpaintCombinedPipeline
    _connected_classes = {"prior_pipe": RBLNKandinskyV22PriorPipeline, "decoder_pipe": RBLNKandinskyV22InpaintPipeline}
    _submodules = ["prior_pipe", "decoder_pipe"]
    _prefix = {"prior_pipe": "prior_"}

    def __init__(
        self,
        unet: "UNet2DConditionModel",
        scheduler: "DDPMScheduler",
        movq: "VQModel",
        prior_prior: "PriorTransformer",
        prior_image_encoder: "CLIPVisionModelWithProjection",
        prior_text_encoder: "CLIPTextModelWithProjection",
        prior_tokenizer: "CLIPTokenizer",
        prior_scheduler: "UnCLIPScheduler",
        prior_image_processor: "CLIPImageProcessor",
    ):
        RBLNDiffusionMixin.__init__(self)
        super(KandinskyV22InpaintCombinedPipeline, self).__init__()

        self.register_modules(
            unet=unet,
            scheduler=scheduler,
            movq=movq,
            prior_prior=prior_prior,
            prior_image_encoder=prior_image_encoder,
            prior_text_encoder=prior_text_encoder,
            prior_tokenizer=prior_tokenizer,
            prior_scheduler=prior_scheduler,
            prior_image_processor=prior_image_processor,
        )

        self.prior_pipe = RBLNKandinskyV22PriorPipeline(
            prior=prior_prior,
            image_encoder=prior_image_encoder,
            text_encoder=prior_text_encoder,
            tokenizer=prior_tokenizer,
            scheduler=prior_scheduler,
            image_processor=prior_image_processor,
        )
        self.decoder_pipe = RBLNKandinskyV22InpaintPipeline(
            unet=unet,
            scheduler=scheduler,
            movq=movq,
        )

    def get_compiled_image_size(self):
        return self.movq.image_size
