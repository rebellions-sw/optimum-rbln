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

from diffusers import KandinskyV22Pipeline

from ...modeling_diffusers import RBLNDiffusionMixin


class RBLNKandinskyV22Pipeline(RBLNDiffusionMixin, KandinskyV22Pipeline):
    original_class = KandinskyV22Pipeline
    _submodules = ["unet", "movq"]

    def get_compiled_image_size(self):
        return self.movq.image_size

    def validate_model_runtime_consistency(self, *args, **kwargs):
        if self.movq.compiled_batch_size == self.unet.compiled_batch_size:
            do_classifier_free_guidance = False
        elif self.movq.compiled_batch_size * 2 == self.unet.compiled_batch_size:
            do_classifier_free_guidance = True
        else:
            raise ValueError(
                "Inconsistent batch sizes between `unet` and `movq`. "
                f"`unet` batch size: {self.unet.compiled_batch_size}, "
                f"`movq` batch size: {self.movq.compiled_batch_size}. "
                "The batch size of `unet` must be either equal to or twice the batch size of `movq`."
            )
        guidance_scale = kwargs.get("guidance_scale", 5.0)
        if not ((guidance_scale <= 1.0) ^ do_classifier_free_guidance):
            raise ValueError(
                f"`guidance_scale` ({guidance_scale}) is incompetible with the compiled batch sizes of `unet` and `movq`. "
                f"Those models are compiled assuming that classifier-free guidance is {'enabled' if do_classifier_free_guidance else 'disabled'}. "
                "Please ensure `guidance_scale` is > 1.0 when classifier-free guidance is enabled, and <= 1.0 otherwise."
            )
