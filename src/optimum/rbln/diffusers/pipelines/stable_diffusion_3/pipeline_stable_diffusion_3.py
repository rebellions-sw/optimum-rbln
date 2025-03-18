# Copyright 2025 Rebellions Inc. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from diffusers import StableDiffusion3Pipeline

from ...modeling_diffusers import RBLNDiffusionMixin


class RBLNStableDiffusion3Pipeline(RBLNDiffusionMixin, StableDiffusion3Pipeline):
    original_class = StableDiffusion3Pipeline
    _submodules = ["transformer", "text_encoder_3", "text_encoder", "text_encoder_2", "vae"]

    def validate_model_runtime_consistency(self, *args, **kwargs):
        if self.vae.compiled_batch_size == self.transformer.compiled_batch_size:
            do_classifier_free_guidance = False
        elif self.vae.compiled_batch_size * 2 == self.transformer.compiled_batch_size:
            do_classifier_free_guidance = True
        else:
            raise ValueError("The batch size of `transformer` must be either equal to or twice the batch size of `vae`.")
        guidance_scale = kwargs.get("guidance_scale", 5.0)
        if not ((guidance_scale <= 1.) ^ do_classifier_free_guidance):
            raise ValueError("`guidance_scale` is not competible with compiled batch sizes of `transformer` and `vae`.")
