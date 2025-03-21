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

from diffusers.pipelines.pipeline_utils import ALL_IMPORTABLE_CLASSES, LOADABLE_CLASSES
from transformers.utils import _LazyModule


LOADABLE_CLASSES["optimum.rbln"] = {"RBLNBaseModel": ["save_pretrained", "from_pretrained"]}
ALL_IMPORTABLE_CLASSES.update(LOADABLE_CLASSES["optimum.rbln"])


_import_structure = {
    "pipelines": [
        "RBLNKandinskyV22InpaintCombinedPipeline",
        "RBLNKandinskyV22InpaintPipeline",
        "RBLNKandinskyV22PriorPipeline",
        "RBLNStableDiffusionPipeline",
        "RBLNStableDiffusionXLPipeline",
        "RBLNStableDiffusionImg2ImgPipeline",
        "RBLNStableDiffusionInpaintPipeline",
        "RBLNStableDiffusionControlNetImg2ImgPipeline",
        "RBLNMultiControlNetModel",
        "RBLNStableDiffusionXLImg2ImgPipeline",
        "RBLNStableDiffusionXLInpaintPipeline",
        "RBLNStableDiffusionControlNetPipeline",
        "RBLNStableDiffusionXLControlNetPipeline",
        "RBLNStableDiffusionXLControlNetImg2ImgPipeline",
        "RBLNStableDiffusion3Pipeline",
        "RBLNStableDiffusion3Img2ImgPipeline",
        "RBLNStableDiffusion3InpaintPipeline",
    ],
    "models": [
        "RBLNAutoencoderKL",
        "RBLNUNet2DConditionModel",
        "RBLNControlNetModel",
        "RBLNSD3Transformer2DModel",
        "RBLNPriorTransformer",
        "RBLNVQModel",
    ],
    "modeling_diffusers": [
        "RBLNDiffusionMixin",
    ],
}

if TYPE_CHECKING:
    from .modeling_diffusers import RBLNDiffusionMixin
    from .models import (
        RBLNAutoencoderKL,
        RBLNControlNetModel,
        RBLNPriorTransformer,
        RBLNSD3Transformer2DModel,
        RBLNUNet2DConditionModel,
        RBLNVQModel,
    )
    from .pipelines import (
        RBLNKandinskyV22InpaintCombinedPipeline,
        RBLNKandinskyV22InpaintPipeline,
        RBLNKandinskyV22PriorPipeline,
        RBLNMultiControlNetModel,
        RBLNStableDiffusion3Img2ImgPipeline,
        RBLNStableDiffusion3InpaintPipeline,
        RBLNStableDiffusion3Pipeline,
        RBLNStableDiffusionControlNetImg2ImgPipeline,
        RBLNStableDiffusionControlNetPipeline,
        RBLNStableDiffusionImg2ImgPipeline,
        RBLNStableDiffusionInpaintPipeline,
        RBLNStableDiffusionPipeline,
        RBLNStableDiffusionXLControlNetImg2ImgPipeline,
        RBLNStableDiffusionXLControlNetPipeline,
        RBLNStableDiffusionXLImg2ImgPipeline,
        RBLNStableDiffusionXLInpaintPipeline,
        RBLNStableDiffusionXLPipeline,
    )
else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
    )
