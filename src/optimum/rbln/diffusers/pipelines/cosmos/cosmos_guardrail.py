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
import json
import os
import pathlib
import re
import string
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union, Iterable

import rebel
import torch  # noqa: I001

import numpy as np
from abc import ABC, abstractmethod
import PIL.Image
from diffusers.pipelines.cosmos.cosmos_guardrail import (
    CosmosSafetyChecker, 
    RetinaFaceFilter,
    TensorDataset,
    DataLoader,
    PriorBox,
    GuardrailRunner,
    VideoContentSafetyFilter,
    SigLIPEncoder,
    Aegis)

from optimum.rbln import RBLNLlamaForCausalLM
from pytorch_retinaface.data import cfg_re50
from diffusers.models.modeling_outputs import AutoencoderKLOutput
from transformers import AutoConfig, PretrainedConfig, SiglipProcessor

from ....modeling import RBLNModel, RBLNBaseModel
from ....modeling_config import DEFAULT_COMPILED_MODEL_NAME, RBLNCompileConfig, RBLNConfig, use_rbln_config
from ....utils.hub import validate_files
from ....utils.logging import get_logger
from ....utils.runtime_utils import UnavailableRuntime
from ...modeling_diffusers import RBLNDiffusionMixin
from ...modeling_pt import RBLNTempModule
from ....utils.runtime_utils import RBLNPytorchRuntime
# from .vae import RBLNRuntimeVAEDecoder, RBLNRuntimeVAEEncoder, _VAEDecoder, _VAEEncoder

# from pytorch_retinaface.models.retinaface import RetinaFace

from diffusers.utils import (
    get_logger,
    is_better_profanity_available,
    is_nltk_available,
    is_peft_available,
    is_pytorch_retinaface_available,
    load_video,
)

from diffusers.pipelines.cosmos.cosmos_utils import (
    CLASS_IDX_TO_NAME,
    KEEP_TOP_K,
    NMS_THRESHOLD,
    TOP_K,
    UNSAFE_CATEGORIES,
    decode_batch,
    filter_detected_boxes,
    load_model,
    pixelate_face,
    read_keyword_list_from_dir,
    to_ascii,
)



if TYPE_CHECKING:
    import torch
    from transformers import AutoFeatureExtractor, AutoProcessor, AutoTokenizer, PretrainedConfig

logger = get_logger(__name__)

COSMOS_GUARDRAIL_CHECKPOINT = "nvidia/Cosmos-1.0-Guardrail"   

class RBLNsimpleModel:
    def __init__(
        self,
        models: List[rebel.Runtime],
        rbln_config: RBLNConfig,
        model_save_dir: Optional[Union[str, Path, TemporaryDirectory]] = None,
        rbln_compiled_models: Optional[rebel.RBLNCompiledModel] = None,
        subfolder: str = "",
        ):
        self.model = models
        self.rbln_config = rbln_config
        self.compiled_models = rbln_compiled_models
        self.model_save_dir = model_save_dir
        self.subfolder = subfolder
    
    @classmethod
    def load_compiled_models(cls, model_id, rbln_config, subfolder=""):
        return cls._load_compiled_models(model_id=model_id, subfolder=subfolder, rbln_config=rbln_config)
    
    @classmethod
    def _load_compiled_models(cls, model_id: str, subfolder: str, rbln_config, rbln_compiled_models=None):
        model_path = Path(model_id)
        
        model_save_dir = model_path / subfolder
        rbln_files = list(model_save_dir.glob("*.rbln"))
        rbln_config_filenames = list(model_save_dir.glob("rbln_config.json"))
        validate_files(rbln_files, rbln_config_filenames, f"directory {model_save_dir}")
        
        from_export_method = isinstance(rbln_config, RBLNConfig) and rbln_compiled_models is not None
        if not from_export_method:
            rbln_kwargs = rbln_config or {}
            rbln_config = RBLNConfig.load(model_save_dir)
            rbln_config.update_runtime_cfg(rbln_kwargs)
        
        compiled_models = Path(model_save_dir).glob("*.rbln")
        rbln_compiled_models = {cm.stem: rebel.RBLNCompiledModel(cm) for cm in compiled_models}
        return cls._from_compiled_models(rbln_compiled_models, rbln_config, model_path=model_path, subfolder=subfolder)
    
    @classmethod
    def _from_compiled_models(
        cls,
        rbln_compiled_models: Dict[str, rebel.RBLNCompiledModel],
        rbln_config: RBLNConfig,
        model_save_dir: Optional[Union[str, Path, TemporaryDirectory]] = None,
        subfolder = "",
        **kwargs,
    ):
        compiled_model_names = [cfg.compiled_model_name for cfg in rbln_config.compile_cfgs]
        rbln_compiled_models = [rbln_compiled_models[cm_name] for cm_name in compiled_model_names]
        # create runtimes only if `rbln_create_runtimes` is enabled
        try:
            models = (
                cls._create_runtimes(rbln_compiled_models, rbln_config.device_map, rbln_config.activate_profiler)
                if rbln_config.create_runtimes
                else UnavailableRuntime()
            )

        except rebel.core.exception.RBLNRuntimeError as e:
            logger.warning(
                f"Failed to create the runtime for the model due to a runtime error: {e.__class__.__name__} - {e}"
            )
            models = UnavailableRuntime()

        models = [cls.wrap_runtime_if_needed(model) for model in models]
        
        return cls(models,
                   rbln_config=rbln_config,
                    model_save_dir= model_save_dir,
                    rbln_compiled_models=rbln_compiled_models,
                    subfolder=subfolder
                   )
    
    @classmethod
    def wrap_runtime_if_needed(cls, model):
        return model
    
    @classmethod
    def wrap_model_if_needed(cls, model):
        return model
    
    @classmethod
    def compile_model(cls, model, rbln_kwargs, model_save_dir, subfolder=""):
        rbln_config = cls._get_rbln_config(rbln_kwargs)
        compiled_model = cls._get_compiled_model(model, rbln_config)
        
        # Save compiled models (.rbln)
        (model_save_dir / subfolder).mkdir(exist_ok=True)
        if not isinstance(compiled_model, dict):
            compiled_models = {DEFAULT_COMPILED_MODEL_NAME: compiled_model}
        else:
            compiled_models = compiled_model
        for compiled_model_name, cm in compiled_models.items():
            cm.save(model_save_dir / subfolder / f"{compiled_model_name}.rbln")
        
        rbln_config.save(model_save_dir / subfolder)
        # Save torch artifacts (e.g. embedding matrix if needed.)
        # cls.save_torch_artifacts(model, save_dir_path=save_dir_path, subfolder=subfolder, rbln_config=rbln_config)
        
        rbln_compiled_models = cls._load_compiled_models(model_id=model_save_dir, subfolder=subfolder, rbln_config=rbln_config, rbln_compiled_models=compiled_models)
        return rbln_compiled_models
    
    @classmethod
    def _get_rbln_config(cls, rbln_kwargs):
        pass
        input_info = [("frames", [1, 1, 1, 1], "float32")]
        
        compile_config = RBLNCompileConfig(
            input_info=input_info
        )
        
        rbln_config = RBLNConfig(
            rbln_cls=cls.__name__,
            compile_cfgs=[compile_config],
            rbln_kwargs=rbln_kwargs,
        )
        return rbln_config
    
    @classmethod
    def _get_compiled_model(cls, model, rbln_config):
        return cls.compile(cls.wrap_model_if_needed(model), rbln_compile_config=rbln_config.compile_cfgs[0])

    @classmethod
    def compile(cls, model, rbln_compile_config: Optional[RBLNCompileConfig] = None, **kwargs):
        compiled_model = rebel.compile_from_torch(
            model,
            input_info=rbln_compile_config.input_info,
            fusion=rbln_compile_config.fusion,
            npu=rbln_compile_config.npu,
            tensor_parallel_size=rbln_compile_config.tensor_parallel_size,
            **kwargs,
        )
        return compiled_model
    
    @classmethod
    def _create_runtimes(
        cls,
        compiled_models: List[rebel.RBLNCompiledModel],
        rbln_device_map: Dict[str, int],
        activate_profiler: Optional[bool] = None,
    ) -> List[rebel.Runtime]:
        if DEFAULT_COMPILED_MODEL_NAME not in rbln_device_map:
            cls._raise_missing_compiled_file_error([DEFAULT_COMPILED_MODEL_NAME])

        device = rbln_device_map[DEFAULT_COMPILED_MODEL_NAME]
        return [
            compiled_model.create_runtime(tensor_type="pt", device=device, activate_profiler=activate_profiler)
            for compiled_model in compiled_models
        ]
    
    @staticmethod
    def _raise_missing_compiled_file_error(missing_files: List[str]):
        """Raises a KeyError with a message indicating missing compiled model files."""

        if len(missing_files) == 1:
            message = f"The rbln model folder is missing the required '{missing_files[0]}.rbln' file. "
        else:
            files_str = ", ".join([f"'{f}.rbln'" for f in missing_files])
            message = (
                "The rbln model folder is missing required files. "
                f"Ensure that {files_str} files are present in the folder. "
            )
        message += (
            "These files are necessary for loading the rbln model. "
            "If these files are missing, please recompile the model using the latest optimum-rbln "
            "and ensure the compilation completes successfully."
        )
        raise KeyError(message)
        
    def parameters(self):
        for param in [torch.tensor([1], dtype=torch.float32, device=torch.device("cpu"))]:
            yield param
    
    def __call__(self, *args: List[torch.Tensor], **kwargs: Dict[str, torch.Tensor]):
        output = self.model[0](*args, **kwargs)
        return output

class RBLNRetinaFace(RBLNsimpleModel):
    @classmethod
    def _get_rbln_config(cls, rbln_kwargs):
        height = rbln_kwargs.get("height", 704)
        width = rbln_kwargs.get("width", 1280)
        input_info = [("frames", [1, 3, height, width], "float32")] # hard coded
        
        postprocessor_config = RBLNCompileConfig(
            input_info=input_info
        )
        
        rbln_config = RBLNConfig(
            rbln_cls=cls.__name__,
            compile_cfgs=[postprocessor_config],
            rbln_kwargs=rbln_kwargs,
        )
        return rbln_config

class RBLNSafetyClassifier(RBLNsimpleModel):
    @classmethod
    def _get_rbln_config(cls, rbln_kwargs):
        input_info_cls = [("data", [1, 1152], "float32")]                # hard coded
        cls_config = RBLNCompileConfig(
            input_info=input_info_cls
        )

        rbln_config = RBLNConfig(
            rbln_cls=cls.__name__,
            compile_cfgs=[cls_config],
            rbln_kwargs=rbln_kwargs,
        )
        return rbln_config

class RBLNVideoSafetyModel:
    def __init__(self, model):
        self.network = model
    
    def parameters(self):
        return self.network.parameters()

# class RBLNVideoSafetyModel(RBLNsimpleModel):
#     @classmethod
#     def _get_rbln_config(cls, rbln_kwargs):
#         input_info_cls = [("data", [1, 1152], "float32")]                # hard coded
#         cls_config = RBLNCompileConfig(
#             input_info=input_info_cls
#         )

#         rbln_config = RBLNConfig(
#             rbln_cls=cls.__name__,
#             compile_cfgs=[cls_config],
#             rbln_kwargs=rbln_kwargs,
#         )
#         return rbln_config
    
#     @classmethod
#     def wrap_model_if_needed(cls, model):
#         return model.network
    
#     def network(self, x):
#         return self.forward(x)
    
#     # def parameters(self):
#     #     return self.network.parameters()

class RBLNRuntimeSiglipVisionModel(RBLNPytorchRuntime):
    def __init__(self, runtime, **kwargs):
        super().__init__(runtime=runtime, **kwargs)
        self.model = runtime
        
    def get_image_features(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        interpolate_pos_encoding: bool = False,
        ) -> torch.FloatTensor:
        # Use SiglipModel's config for some fields (if specified) instead of those of vision & text components.
        vision_outputs = self.model(
            pixel_values=pixel_values,
        )
        pooled_output = vision_outputs[1]
        return pooled_output

class _SiglipVisionModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values):
        return self.model(
            pixel_values=pixel_values,
            return_dict=False,
        )

class RBLNSiglipVisionModel(RBLNsimpleModel):
    @classmethod
    def _get_rbln_config(cls, rbln_kwargs):
        input_info_enc = [("pixel_values", [1, 3, 384, 384], "float32")] # hard coded
        enc_config = RBLNCompileConfig(
            input_info=input_info_enc
        )

        rbln_config = RBLNConfig(
            rbln_cls=cls.__name__,
            compile_cfgs=[enc_config],
            rbln_kwargs=rbln_kwargs,
        )
        return rbln_config
    
    @classmethod
    def wrap_model_if_needed(cls, model):
        return _SiglipVisionModel(model)
    
    @classmethod
    def wrap_runtime_if_needed(cls, model):
        return RBLNRuntimeSiglipVisionModel(model)


class RBLNLlamaGuard(RBLNLlamaForCausalLM):
    @classmethod
    def load_compiled_models(cls, model_id, rbln_config, subfolder=""):
        text_config = rbln_config.get("text_guardrail", {})
        model = RBLNLlamaForCausalLM.from_pretrained(
            model_id=model_id,
            export=False,
            rbln_config=text_config,
        )
        return model
        
    @classmethod
    def compile_model(cls, model, rbln_kwargs, model_save_dir, subfolder=""):
        batch_size = rbln_kwargs.get("batch_size", 1)
        
        max_seq_len = rbln_kwargs.get("max_seq_len", 4096)
        tensor_parallel_size = rbln_kwargs.get("tensor_parallel_size", 4)
        model = model.merge_and_unload()
        compiled_model = RBLNLlamaForCausalLM.from_model(
                    model, 
                    export=True,
                    rbln_batch_size=batch_size,
                    rbln_max_seq_len=max_seq_len,
                    rbln_tensor_parallel_size=tensor_parallel_size,
                    model_save_dir=model_save_dir,
                    subfolder=subfolder
                    )
        return compiled_model

class RBLNCosmosSafetyChecker:
    original_class = CosmosSafetyChecker
    # _submodules = {
    #     "video_guardrail" : 
    #         {
    #             "postprocessors" : {
    #                 0 : {
    #                     "net" :RBLNRetinaFace
    #                     }
    #                 },
    #             "safety_models" : {
    #                 0: {
    #                     "encoder" : {
    #                         "model":
    #                             {
    #                                 "vision_model": RBLNSiglipVisionModel 
    #                                 }
    #                         },
    #                     "model" : {
    #                         "network" : RBLNSafetyClassifier
    #                         }
    #                     }
    #                 },
    #             },
    #     "text_guardrail": 
    #         {
    #             "safety_models" : {
    #                 1 : {
    #                     "model" : RBLNLlamaForCausalLM
    #                     }
    #                 }
    #             }, 
    #     }
    _submodules_dir = {
        "video_guardrail" : 
                {
                    "postprocessors": "net",   
                    "safety_models": {
                        "encoder": "vision_model", 
                        "model": "network"
                        }
                    },
        "text_guardrail": "safety_models"
        }
    
    @classmethod
    @use_rbln_config
    def compile_submodules(cls, model, rbln_config, model_save_dir:str="safety_checker", subfolder:str=""):
        # model_save_dir = Path(model_save_dir)
        # model_save_dir.mkdir(exist_ok=True)
        
        # Todo Parsing
        
        # for guardrail, targets in cls._submodules.items():
        #     for target, ids in targets.items():
        #         target_path = Path(model_save_dir/guardrail/target)
        #         for id, rbln_module in ids.items():
        #             for module_name, _rbln_module in rbln_module.items():
        #                 target_model = getattr(getattr(getattr(model, guardrail), target)[id], module_name)
        #                 module_path = Path(target_path/module_name)
        #                 os.makedirs(module_path, exist_ok=True)
                        
        #                 if isinstance(_rbln_module, dict):
        #                     for _name, _module in _rbln_module.items():
        #                         target_model = getattr(target_model, _name)
        #                         _module_path = Path(module_path/_name)
        #                         os.makedirs(_module_path, exist_ok=True)
        #                         if isinstance(_module, dict):
        #                             for n, m in _module.items():
        #                                 target_model = getattr(target_model, n)
        #                                 compiled_model = m.compile_model(
        #                                     model=target_model,
        #                                     rbln_kwargs=rbln_config,
        #                                     model_save_dir=_module_path,
        #                                     subfolder=n
        #                                     )
        #                         else :
        #                             compiled_model = _rbln_module.compile_model(
        #                             model=target_model,
        #                             rbln_kwargs=rbln_config,
        #                             model_save_dir=_module_path,
        #                             subfolder=subfolder
        #                             )
        #                 else :
        #                     compiled_model = _rbln_module.compile_model(
        #                         model=target_model,
        #                         rbln_kwargs=rbln_config,
        #                         model_save_dir=module_path,
        #                         subfolder=subfolder
        #                         )
                            
        
        
        
        save_dir_path = Path(model_save_dir)
        save_dir_path.mkdir(exist_ok=True)
        
        save_dir_path = Path(model_save_dir+"/video_guardrail/postprocessors")
        os.makedirs(save_dir_path, exist_ok=True)
        
        model_name = "net"
        _model = model.video_guardrail.postprocessors[0]
        compile_target = getattr(_model, model_name)
        compiled_model = RBLNRetinaFace.compile_model(compile_target,
                                                rbln_kwargs=rbln_config["video_guardrail"],
                                                model_save_dir=save_dir_path,
                                                )
        delattr(_model, model_name)
        setattr(_model, model_name, compiled_model)
        
        save_dir_path = Path(model_save_dir+"/video_guardrail/safety_models")
        os.makedirs(save_dir_path, exist_ok=True)
        
        model_name = "network"
        _model = model.video_guardrail.safety_models[0].model
        compile_target = getattr(_model, model_name)
        compiled_model = RBLNSafetyClassifier.compile_model(compile_target,
                                                rbln_kwargs=rbln_config["video_guardrail"],
                                                model_save_dir=save_dir_path,
                                                subfolder=f"model",
                                                )
        delattr(model.video_guardrail.safety_models[0], "model")
        setattr(model.video_guardrail.safety_models[0], "model", RBLNVideoSafetyModel(compiled_model))
        
        # model_name = "model"
        # _model = model.video_guardrail.safety_models[0]
        # compile_target = getattr(_model, model_name)
        # compiled_model = RBLNVideoSafetyModel.compile_model(compile_target,
        #                                         rbln_kwargs=rbln_config["video_guardrail"],
        #                                         model_save_dir=save_dir_path,
        #                                         subfolder=f"model",
        #                                         )
        # delattr(_model, model_name)
        # setattr(_model, model_name, compiled_model)
        
        model_name = "vision_model"
        _model = model.video_guardrail.safety_models[0].encoder.model
        compile_target = getattr(_model, model_name)
        compiled_model = RBLNSiglipVisionModel.compile_model(compile_target,
                                                rbln_kwargs=rbln_config["video_guardrail"],
                                                model_save_dir=save_dir_path,
                                                subfolder=f"encoder",
                                                )
        delattr(_model, model_name)
        setattr(_model, model_name, compiled_model)
        
        
        save_dir_path = Path(model_save_dir+"/text_guardrail/safety_models")
        os.makedirs(save_dir_path, exist_ok=True)
        
        model_name = "model"
        _model = model.text_guardrail.safety_models[1]
        compile_target = getattr(_model, model_name)
        compiled_model = RBLNLlamaGuard.compile_model(
                                                compile_target,
                                                rbln_kwargs=rbln_config["text_guardrail"],
                                                model_save_dir=save_dir_path,
                                                )
        delattr(_model, model_name)
        setattr(_model, model_name, compiled_model)
        return model

    @classmethod
    @use_rbln_config
    def load_submodules(cls, model, rbln_config, subfolder="", model_save_dir=""):
        model_name = "net"
        _model = model.video_guardrail.postprocessors[0]
        save_dir_path = Path(model_save_dir+"/video_guardrail/postprocessors")
        compiled_model = RBLNRetinaFace.load_compiled_models(
            save_dir_path,
            rbln_config=rbln_config["video_guardrail"],
            )
        delattr(_model, model_name)
        setattr(_model, model_name, compiled_model)
        
        model_name = "network"
        _model = model.video_guardrail.safety_models[0].model
        save_dir_path = Path(model_save_dir+"/video_guardrail/safety_models")
        compiled_model = RBLNSafetyClassifier.load_compiled_models(
            save_dir_path,
            rbln_config=rbln_config["video_guardrail"],
            subfolder=f"model",
            )
        delattr(model.video_guardrail.safety_models[0], "model")
        setattr(model.video_guardrail.safety_models[0], "model", RBLNVideoSafetyModel(compiled_model))
        
        model_name = "vision_model"
        _model = model.video_guardrail.safety_models[0].encoder.model
        compiled_model = RBLNSiglipVisionModel.load_compiled_models(
            save_dir_path,
            rbln_config=rbln_config["video_guardrail"],
            subfolder=f"encoder",
            )
        delattr(_model, model_name)
        setattr(_model, model_name, compiled_model)
        
        
        save_dir_path = Path(model_save_dir+"/text_guardrail/safety_models")
        model_name = "model"
        _model = model.text_guardrail.safety_models[1]
        compiled_model = RBLNLlamaGuard.load_compiled_models(
            save_dir_path,
            rbln_config=rbln_config["text_guardrail"],
        )
        delattr(_model, model_name)
        setattr(_model, model_name, compiled_model)
        return model

    def check_text_safety(self, prompt: str) -> bool:
        is_safe, message = self.text_guardrail.run_safety_check(prompt)
        if not is_safe:
            logger.critical(f"GUARDRAIL BLOCKED: {message}")
        return is_safe

    def check_video_safety(self, frames: np.ndarray) -> np.ndarray:
        # for test ; frames = torch.randint(0,255, (121, 1280, 704, 3), dtype=torch.uint8).numpy()
        is_safe, message = self.video_guardrail.run_safety_check(frames)
        if not is_safe:
            logger.critical(f"GUARDRAIL BLOCKED: {message}")
            return None
        frames = self.video_guardrail.postprocess(frames)
        return frames