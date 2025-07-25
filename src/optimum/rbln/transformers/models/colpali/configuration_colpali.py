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
from typing import Any, List, Optional, Union

from ....configuration_utils import RBLNModelConfig


class RBLNColPaliForRetrievalConfig(RBLNModelConfig):
    """
    Configuration class for RBLN ColPali models for document retrieval.

    This class extends RBLNModelConfig with specific configurations for ColPali models,
    including vision tower settings and multi-sequence length support.

    Example usage:
    ```python
    from optimum.rbln import RBLNColPaliForRetrieval, RBLNColPaliForRetrievalConfig

    # Create a configuration object
    config = RBLNColPaliForRetrievalConfig(
        max_seq_lens=1152,
        output_hidden_states=False,
        tensor_parallel_size=4
    )

    # Use the configuration with from_pretrained
    model = RBLNColPaliForRetrieval.from_pretrained(
        "vidore/colpali-v1.3-hf",
        export=True,
        rbln_config=config
    )
    ```
    """

    submodules = ["vision_tower"]

    def __init__(
        self,
        max_seq_lens: Union[int, List[int]] = None,
        output_hidden_states: Optional[bool] = None,
        vision_tower: Optional[RBLNModelConfig] = None,
        **kwargs: Any,
    ):
        """
        Args:
            vision_tower (Optional[RBLNModelConfig]): Configuration for the vision encoder component.
            max_seq_lens (Union[int, List[int]]): The maximum sequence lengths for the language model.
                This can be multiple values, and the model will be compiled for each max_seq_len, allowing selection of the most appropriate max_seq_len at inference time.
            output_hidden_states (Optional[bool]): Whether to output the hidden states of the language model.
            **kwargs: Additional arguments passed to the parent RBLNModelConfig.
        Raises:
            ValueError: If batch_size is not a positive integer.
        """
        super().__init__(**kwargs)
        self.vision_tower = vision_tower
        self.max_seq_lens = max_seq_lens
        self.output_hidden_states = output_hidden_states
