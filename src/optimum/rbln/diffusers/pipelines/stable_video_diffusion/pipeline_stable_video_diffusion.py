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

# Portions of this software are licensed under the Apache License,
# Version 2.0. See the NOTICE file distributed with this work for
# additional information regarding copyright ownership.

# All other portions of this software, including proprietary code,
# are the intellectual property of Rebellions Inc. and may not be
# copied, modified, or distributed without prior written permission
# from Rebellions Inc.
"""RBLNStableVideoDiffusionPipeline class for inference of video diffusion models on rbln devices."""

from typing import Union

import torch
from diffusers import StableVideoDiffusionPipeline
from diffusers.image_processor import PipelineImageInput
from diffusers.pipelines.stable_video_diffusion.pipeline_stable_video_diffusion import _resize_with_antialiasing

from ....utils.logging import get_logger
from ...modeling_diffusers import RBLNDiffusionMixin


logger = get_logger(__name__)


class RBLNStableVideoDiffusionPipeline(RBLNDiffusionMixin, StableVideoDiffusionPipeline):
    """
    Pipeline for image-to-video generation using Stable Video Diffusion.

    This model inherits from [`StableVideoDiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    It implements the methods to convert a pre-trained Stable Video Diffusion pipeline into a RBLNStableVideoDiffusion pipeline by:
    - transferring the checkpoint weights of the original into an optimized RBLN graph,
    - compiling the resulting graph using the RBLN compiler.
    """

    original_class = StableVideoDiffusionPipeline
    _submodules = ["image_encoder", "unet", "vae"]

    def handle_additional_kwargs(self, **kwargs):
        compiled_num_frames = self.unet.rbln_config.model_cfg.get("num_frames")
        if compiled_num_frames is not None:
            kwargs["num_frames"] = compiled_num_frames

        compiled_decode_chunk_size = self.vae.rbln_config.model_cfg.get("decode_chunk_size")
        if compiled_decode_chunk_size is not None:
            kwargs["decode_chunk_size"] = compiled_decode_chunk_size
        return kwargs

    def _encode_image(
        self,
        image: PipelineImageInput,
        device: Union[str, torch.device],
        num_videos_per_prompt: int,
        do_classifier_free_guidance: bool,
    ) -> torch.Tensor:
        dtype = self.image_encoder.dtype

        if not isinstance(image, torch.Tensor):
            image = self.video_processor.pil_to_numpy(image)
            image = self.video_processor.numpy_to_pt(image)

            # We normalize the image before resizing to match with the original implementation.
            # Then we unnormalize it after resizing.
            image = image * 2.0 - 1.0
            image = _resize_with_antialiasing(image, (224, 224))
            image = (image + 1.0) / 2.0

        # Normalize the image with for CLIP input
        image = self.feature_extractor(
            images=image,
            do_normalize=True,
            do_center_crop=False,
            do_resize=False,
            do_rescale=False,
            return_tensors="pt",
        ).pixel_values

        image = image.to(device=device, dtype=dtype)
        image_embeddings = self.image_encoder(image).image_embeds
        image_embeddings = image_embeddings.unsqueeze(1)

        # duplicate image embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = image_embeddings.shape
        image_embeddings = image_embeddings.repeat(1, num_videos_per_prompt, 1)
        image_embeddings = image_embeddings.view(bs_embed * num_videos_per_prompt, seq_len, -1)

        if do_classifier_free_guidance:
            negative_image_embeddings = torch.zeros_like(image_embeddings)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            image_embeddings = torch.cat([negative_image_embeddings, image_embeddings])

        return image_embeddings
