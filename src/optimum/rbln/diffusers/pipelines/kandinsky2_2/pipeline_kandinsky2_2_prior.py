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

import inspect
from diffusers import KandinskyV22PriorPipeline

from ...modeling_diffusers import RBLNDiffusionMixin


class RBLNKandinskyV22PriorPipeline(RBLNDiffusionMixin, KandinskyV22PriorPipeline):
    original_class = KandinskyV22PriorPipeline
    _submodules = ["text_encoder", "image_encoder", "prior"]

    def validate_model_runtime_consistency(self, *args, **kwargs):
        param_names = list(inspect.signature(self.original_class.__call__).parameters.keys())[1:]
        args_dict = dict(zip(param_names, args))
        merged = {**args_dict, **kwargs}
        prompt = merged.get("prompt")
        negative_prompt = merged.get("negative_prompt", None)
        guidance_scale = merged.get("guidance_scale", 5.0)
        batch_size = len(prompt) if isinstance(prompt, list) else 1
        if self.prior.compiled_batch_size == self.text_encoder.compiled_batch_size:
            do_classifier_free_guidance = False
        elif self.prior.compiled_batch_size == self.text_encoder.compiled_batch_size * 2:
            do_classifier_free_guidance = True
        else:
            raise ValueError("The batch size of `prior` must be either equal to or twice the batch size of `text_encoder`.")

        if negative_prompt is not None:
            if self.text_encoder.compiled_batch_size != batch_size * 2:
                raise ValueError("If `negative_prompt` is provided, the compiled batch size of `text_encoder` should be double compared to batch size")

        if not ((guidance_scale <= 1.) ^ do_classifier_free_guidance):
            raise ValueError("`guidance_scale` is not competible with compiled batch sizes of `text_encoder` and `prior`.")

