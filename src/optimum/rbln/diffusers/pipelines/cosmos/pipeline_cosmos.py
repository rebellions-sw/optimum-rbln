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


from inspect import signature

from diffusers import CosmosPipeline

from ...modeling_diffusers import RBLNDiffusionMixin


class RBLNCosmosPipeline(RBLNDiffusionMixin, CosmosPipeline):
    original_class = CosmosPipeline
    _submodules = ["text_encoder", "transformer", "vae", "safety_checker"]

    def forward(self, *args, **kwargs):
        param_names = list(signature(super().forward).parameters.keys())
        for i, value in enumerate(args):
            kwargs[param_names[i]] = value

        num_inference_steps = kwargs["num_inference_steps"]
        if (not hasattr(self.transformer, "_emb_cached")) or len(self.transformer._emb_cached) != num_inference_steps:
            self.transformer.get_time_embed_table(self.scheduler, num_inference_steps)
        super().forward(**kwargs)
