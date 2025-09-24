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

from typing import Any, Optional

from ....configuration_utils import RBLNModelConfig
from ....transformers import RBLNT5EncoderModelConfig
from ....utils.logging import get_logger
from ...pipelines.cosmos.cosmos_guardrail import RBLNCosmosSafetyCheckerConfig
from ..models import RBLNAutoencoderKLCosmosConfig, RBLNCosmosTransformer3DModelConfig


logger = get_logger(__name__)


class RBLNCosmosPipelineBaseConfig(RBLNModelConfig):
    submodules = ["text_encoder", "transformer", "vae", "safety_checker"]
    _vae_uses_encoder = False

    def __init__(
        self,
        text_encoder: Optional[RBLNT5EncoderModelConfig] = None,
        transformer: Optional[RBLNCosmosTransformer3DModelConfig] = None,
        vae: Optional[RBLNAutoencoderKLCosmosConfig] = None,
        safety_checker: Optional[RBLNCosmosSafetyCheckerConfig] = None,
        *,
        batch_size: Optional[int] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_frames: Optional[int] = None,
        fps: Optional[int] = None,
        max_seq_len: Optional[int] = None,
        **kwargs: Any,
    ):
        """
        Args:
            text_encoder (Optional[RBLNT5EncoderModelConfig]): Configuration for the text encoder component.
                Initialized as RBLNT5EncoderModelConfig if not provided.
            transformer (Optional[RBLNCosmosTransformer3DModelConfig]): Configuration for the Transformer model component.
                Initialized as RBLNCosmosTransformer3DModelConfig if not provided.
            vae (Optional[RBLNAutoencoderKLCosmosConfig]): Configuration for the VAE model component.
                Initialized as RBLNAutoencoderKLCosmosConfig if not provided.
            safety_checker (Optional[RBLNCosmosSafetyCheckerConfig]): Configuration for the safety checker component.
                Initialized as RBLNCosmosSafetyCheckerConfig if not provided.
            batch_size (Optional[int]): Batch size for inference, applied to all submodules.
            height (Optional[int]): Height of the generated videos.
            width (Optional[int]): Width of the generated videos.
            num_frames (Optional[int]): The number of frames in the generated video.
            fps (Optional[int]): The frames per second of the generated video.
            max_seq_len (Optional[int]): Maximum sequence length supported by the model.
            **kwargs: Additional arguments passed to the parent RBLNModelConfig.
        """
        super().__init__(**kwargs)

        self.text_encoder = self.init_submodule_config(
            RBLNT5EncoderModelConfig, text_encoder, batch_size=batch_size, max_seq_len=max_seq_len
        )
        self.transformer = self.init_submodule_config(
            RBLNCosmosTransformer3DModelConfig,
            transformer,
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            height=height,
            width=width,
            num_frames=num_frames,
            fps=fps,
        )
        self.vae = self.init_submodule_config(
            RBLNAutoencoderKLCosmosConfig,
            vae,
            batch_size=batch_size,
            uses_encoder=self.__class__._vae_uses_encoder,
            height=height,
            width=width,
            num_frames=num_frames,
        )
        self.safety_checker = self.init_submodule_config(
            RBLNCosmosSafetyCheckerConfig,
            safety_checker,
            batch_size=batch_size,
            height=height,
            width=width,
        )

    @property
    def batch_size(self):
        return self.vae.batch_size

    @property
    def max_seq_len(self):
        return self.text_encoder.max_seq_len


class RBLNCosmosTextToWorldPipelineConfig(RBLNCosmosPipelineBaseConfig):
    """Config for Cosmos Text2World Pipeline"""

    _vae_uses_encoder = False


class RBLNCosmosVideoToWorldPipelineConfig(RBLNCosmosPipelineBaseConfig):
    """Config for Cosmos Video2World Pipeline"""

    _vae_uses_encoder = True
