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

from typing import TYPE_CHECKING

from transformers.utils import _LazyModule


_import_structure = {
    "autoencoders": [
        "RBLNAutoencoderKL",
        "RBLNVQModel",
    ],
    "unets": [
        "RBLNUNet2DConditionModel",
    ],
    "controlnet": ["RBLNControlNetModel"],
    "transformers": [
        "RBLNPriorTransformer",
        "RBLNSD3Transformer2DModel",
    ],
}

if TYPE_CHECKING:
    from .autoencoders import (
        RBLNAutoencoderKL,
        RBLNVQModel,
    )
    from .controlnet import RBLNControlNetModel
    from .transformers import (
        RBLNPriorTransformer,
        RBLNSD3Transformer2DModel,
    )
    from .unets import (
        RBLNUNet2DConditionModel,
    )
else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
    )
