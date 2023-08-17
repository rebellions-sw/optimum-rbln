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

from typing import TYPE_CHECKING

from transformers.utils import _LazyModule


_import_structure = {
    "controlnet": [
        "RBLNMultiControlNetModel",
        "RBLNStableDiffusionControlNetImg2ImgPipeline",
        "RBLNStableDiffusionControlNetPipeline",
        "RBLNStableDiffusionXLControlNetImg2ImgPipeline",
        "RBLNStableDiffusionXLControlNetPipeline",
    ],
    "stable_diffusion": [
        "RBLNStableDiffusionImg2ImgPipeline",
        "RBLNStableDiffusionPipeline",
        "RBLNStableDiffusionInpaintPipeline",
    ],
    "stable_diffusion_xl": [
        "RBLNStableDiffusionXLImg2ImgPipeline",
        "RBLNStableDiffusionXLPipeline",
        "RBLNStableDiffusionXLInpaintPipeline",
    ],
    "stable_diffusion_3": [
        "RBLNStableDiffusion3Pipeline",
        "RBLNStableDiffusion3Img2ImgPipeline",
        "RBLNStableDiffusion3InpaintPipeline",
    ],
}
if TYPE_CHECKING:
    from .controlnet import (
        RBLNMultiControlNetModel,
        RBLNStableDiffusionControlNetImg2ImgPipeline,
        RBLNStableDiffusionControlNetPipeline,
        RBLNStableDiffusionXLControlNetImg2ImgPipeline,
        RBLNStableDiffusionXLControlNetPipeline,
    )
    from .stable_diffusion import (
        RBLNStableDiffusionImg2ImgPipeline,
        RBLNStableDiffusionInpaintPipeline,
        RBLNStableDiffusionPipeline,
    )
    from .stable_diffusion_3 import (
        RBLNStableDiffusion3Img2ImgPipeline,
        RBLNStableDiffusion3InpaintPipeline,
        RBLNStableDiffusion3Pipeline,
    )
    from .stable_diffusion_xl import (
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
