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

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import torch
from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel

from ....modeling import RBLNModel
from ....utils.logging import get_logger
from ...models.controlnet import RBLNControlNetModel


if TYPE_CHECKING:
    pass

logger = get_logger(__name__)


class RBLNMultiControlNetModel(RBLNModel):
    hf_library_name = "diffusers"
    _hf_class = MultiControlNetModel

    def __init__(
        self,
        models: List[RBLNControlNetModel],
    ):
        self.nets = models
        self.dtype = torch.float32

    @property
    def compiled_models(self):
        cm = []
        for net in self.nets:
            cm.extend(net.compiled_models)
        return cm

    @classmethod
    def _from_pretrained(
        cls,
        model_id: Union[str, Path],
        **kwargs,
    ) -> RBLNModel:
        idx = 0
        controlnets = []
        subfolder_name = kwargs.pop("subfolder", None)
        if subfolder_name is not None:
            model_path_to_load = model_id + "/" + subfolder_name
        else:
            model_path_to_load = model_id

        base_model_path_to_load = model_path_to_load

        while os.path.isdir(model_path_to_load):
            controlnet = RBLNControlNetModel.from_pretrained(model_path_to_load, export=False, **kwargs)
            controlnets.append(controlnet)
            idx += 1
            model_path_to_load = base_model_path_to_load + f"_{idx}"

        return cls(
            controlnets,
        )

    def save_pretrained(self, save_directory: Union[str, Path], **kwargs):
        for idx, model in enumerate(self.nets):
            suffix = "" if idx == 0 else f"_{idx}"
            real_save_path = save_directory + suffix
            model.save_pretrained(real_save_path)

    @classmethod
    def _get_rbln_config(cls, **rbln_config_kwargs):
        pass

    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        controlnet_cond: List[torch.Tensor],
        conditioning_scale: List[float],
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guess_mode: bool = False,
        return_dict: bool = True,
    ):
        for i, (image, scale, controlnet) in enumerate(zip(controlnet_cond, conditioning_scale, self.nets)):
            output = controlnet.model[0](
                sample=sample.contiguous(),
                timestep=timestep.float(),
                encoder_hidden_states=encoder_hidden_states,
                controlnet_cond=image,
                conditioning_scale=torch.tensor(scale),
            )

            down_samples, mid_sample = output[:-1], output[-1]

            # merge samples
            if i == 0:
                down_block_res_samples, mid_block_res_sample = down_samples, mid_sample
            else:
                down_block_res_samples = [
                    samples_prev + samples_curr
                    for samples_prev, samples_curr in zip(down_block_res_samples, down_samples)
                ]
                mid_block_res_sample += mid_sample

        return down_block_res_samples, mid_block_res_sample
