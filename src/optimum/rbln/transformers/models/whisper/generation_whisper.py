# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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

"""
Generation utilities for Whisper.
Modified from `transformers.models.whisper.generation_whisper.py`
"""

import torch
import transformers
from packaging import version
from transformers import GenerationMixin
from transformers.models.whisper.generation_whisper import WhisperGenerationMixin


class RBLNWhisperGenerationMixin(WhisperGenerationMixin, GenerationMixin):
    """
    This class is based on transformers version 4.44.2.
    It uses the same generate() method, so it's crucial to maintain the inheritance order.
    Ensure WhisperGenerationMixin is listed before GenerationMixin.
    """

    def _postprocess_outputs(
        self, seek_outputs, decoder_input_ids, return_token_timestamps, generation_config, *args, **kwargs
    ):
        # remove all previously passed decoder input ids
        # should happen only if it is the first generated segment
        start_idx = decoder_input_ids.shape[-1]

        if isinstance(seek_outputs, torch.Tensor):
            return seek_outputs[:, start_idx:], seek_outputs

        if return_token_timestamps and not self.rbln_token_timestamps:
            raise RuntimeError(
                "To use .generate() with return_token_timestamps=True, the model must be compiled with rbln_token_timestamps=True. "
                "You can compile the model by calling .from_pretrained() with export=True and rbln_token_timestamps=True as keyword arguments, "
                "or you can generate with return_token_timestamps=False."
            )

        if return_token_timestamps and hasattr(generation_config, "alignment_heads"):
            num_frames = getattr(generation_config, "num_frames", None)
            if version.parse(transformers.__version__) >= version.parse("4.46.0"):
                seek_outputs["token_timestamps"] = self._extract_token_timestamps(
                    seek_outputs,
                    generation_config.alignment_heads,
                    num_frames=num_frames,
                    num_input_ids=decoder_input_ids.shape[-1],
                )
            else:
                seek_outputs["token_timestamps"] = self._extract_token_timestamps(
                    seek_outputs,
                    generation_config.alignment_heads,
                    num_frames=num_frames,
                )
        seek_outputs["sequences"] = seek_outputs["sequences"][:, start_idx:]

        def split_by_batch_index(values, key, batch_idx):
            if key in ["scores", "encoder_attentions", "encoder_hidden_states", "logits"]:
                return [v[batch_idx].cpu() for v in values]
            if key in ["decoder_attentions", "decoder_hidden_states", "cross_attentions"]:
                return tuple(tuple(w[batch_idx][None].cpu() for w in v) for v in values)
            elif key == "past_key_values":
                # we don't save `past_key_values in rbln
                return None

            return values[batch_idx].cpu()

        sequence_tokens = seek_outputs["sequences"]

        valid_seekoutputs = []
        for k, v in seek_outputs.items():
            if v is not None and len(v) > 0 and v[0] is not None:
                valid_seekoutputs.append((k, v))
        seek_outputs = [
            {k: split_by_batch_index(v, k, i) for k, v in valid_seekoutputs} for i in range(sequence_tokens.shape[0])
        ]

        return sequence_tokens, seek_outputs
