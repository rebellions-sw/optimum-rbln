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

from ....modeling import RBLNModel, RBLNBaseModel, RBLNTorchModel
from ....modeling_config import DEFAULT_COMPILED_MODEL_NAME, RBLNCompileConfig, RBLNConfig, use_rbln_config
from ....utils.logging import get_logger
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

class RBLNVideoContentSafetyFilter(RBLNTorchModel, VideoContentSafetyFilter):
    def __post_init__(self, **kwargs):
        self.encoder = RBLNRuntimeSigLIPEncoder(self.model[0])
        self.cls_model = RBLNPytorchRuntime(self.model[1])
    
    @classmethod
    def wrap_model_if_needed(cls, model: torch.nn.Module, rbln_config) -> torch.nn.Module:
        if rbln_config.compiled_model_name == "safety_models_0":
            return _SiglipVisionModel(model)
        else :
            return model.eval()
        
    @classmethod
    def _get_rbln_config(
        cls,
        preprocessors: Union["AutoFeatureExtractor", "AutoProcessor", "AutoTokenizer"],
        model_config: "PretrainedConfig",
        rbln_kwargs: Dict[str, Any] = {},
    ) -> RBLNConfig:
        input_info_enc = [("pixel_values", [1, 3, 384, 384], "float32")] # hard coded
        input_info_cls = [("data", [1, 1152], "float32")]                # hard coded
        
        compile_cfgs = []
        # for safety_models
        input_infos = [input_info_enc, input_info_cls]
        for i, input_info in enumerate(input_infos):
            safetymodel_config = RBLNCompileConfig(
                compiled_model_name=f"safety_models_{i}",
                input_info=input_info
            )
            compile_cfgs.append(safetymodel_config)

        rbln_config = RBLNConfig(
            rbln_cls=cls.__name__,
            compile_cfgs=compile_cfgs,
            rbln_kwargs=rbln_kwargs,
        )
        return rbln_config
    
    @classmethod
    def get_compiled_model(cls, model, rbln_config: RBLNConfig):
        # SiglipEncoder
        encoder = cls.wrap_model_if_needed(model.encoder.model, rbln_config.compile_cfgs[0])
        encoder_compiled_model = cls.compile(encoder, rbln_compile_config=rbln_config.compile_cfgs[0])

        # classifier
        safetyclassifier = cls.wrap_model_if_needed(model.model.network, rbln_config.compile_cfgs[1])
        safetyclassifier_compiled_model = cls.compile(safetyclassifier, rbln_compile_config=rbln_config.compile_cfgs[1])
        
        return {
            "safety_models_0": encoder_compiled_model,
            "safety_models_1": safetyclassifier_compiled_model,
            }
    
    @classmethod
    def _create_runtimes(
        cls,
        compiled_models: List[rebel.RBLNCompiledModel],
        rbln_device_map: Dict[str, int],
        activate_profiler: Optional[bool] = None,
    ) -> List[rebel.Runtime]:
        if any(model_name not in rbln_device_map for model_name in ["safety_models_0", "safety_models_1"]):
            cls._raise_missing_compiled_file_error(["safety_models_0", "safety_models_1"])

        device_vals = [rbln_device_map["safety_models_0"], rbln_device_map["safety_models_1"]]
        return [
            compiled_model.create_runtime(tensor_type="pt", device=device_val, activate_profiler=activate_profiler)
            for compiled_model, device_val in zip(compiled_models, device_vals)
        ]

    @torch.inference_mode()
    def _infer(self, pil_image: PIL.Image.Image) -> int:
        """Infer the class of the image."""
        image_embs = self.encoder.encode_image(pil_image)
        image_embs = image_embs.to(device=torch.device("cpu"), dtype=torch.float32)
        logits = self.cls_model(image_embs)
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        predicted_class = torch.argmax(probabilities, dim=-1).item()
        return predicted_class

    def is_safe_file(self, filepath: str) -> bool:
        """Check if the video file is safe."""
        video_data = load_video(filepath)

        # Sample frames at 2 FPS
        sample_rate = 2  # frames per second
        frame_interval = int(video_data.fps / sample_rate)
        frame_numbers = list(range(0, int(video_data.fps * video_data.duration), frame_interval))

        is_safe = True
        frame_scores = []

        for frame_number in frame_numbers:
            try:
                frame = video_data.frames[frame_number]
                pil_image = PIL.Image.fromarray(frame)
                predicted_class = self._infer(pil_image)
                class_name = CLASS_IDX_TO_NAME.get(predicted_class, "Unknown")
                frame_scores.append({"frame_number": frame_number, "class": class_name})

                # If any frame is not "Safe", mark the video as unsafe
                if predicted_class != 0:
                    is_safe = False
                    break

            except Exception as e:
                logger.warning(
                    f"Warning: Failed to run safety classifier on frame_number {frame_number}. Exception: {e}"
                )
                continue

        # Prepare data for JSON
        video_data = {
            "filepath": filepath,
            "is_safe": is_safe,
            "video_length": video_data.duration,
            "fps": video_data.fps,
            "frame_scores": frame_scores,
        }

        logger.info(f"Video {filepath} is {'SAFE' if is_safe else 'UNSAFE'}.")
        logger.debug(f"Video data: {json.dumps(video_data, indent=4)}")
        return is_safe

    def is_safe_frames(self, frames: Iterable) -> bool:
        """Check if the video frames are safe."""
        is_safe = True
        frame_scores = []

        for frame_number, frame in enumerate(frames):
            try:
                pil_image = PIL.Image.fromarray(frame)
                predicted_class = self._infer(pil_image)
                class_name = CLASS_IDX_TO_NAME.get(predicted_class, "Unknown")
                frame_scores.append({"frame_number": frame_number, "class": class_name})

                # If any frame is not "Safe", mark as not safe
                if predicted_class != 0:
                    is_safe = False
                    break

            except Exception as e:
                logger.warning(
                    f"Warning: Failed to run safety classifier on frame_number {frame_number}. Exception: {e}"
                )
                continue

        video_data = {
            "is_safe": is_safe,
            "frame_scores": frame_scores,
        }

        logger.debug(f"Frames data: {json.dumps(video_data, indent=4)}")
        return is_safe

class RBLNRetinaFaceFilter(RBLNTorchModel, RetinaFaceFilter):
    def __post_init__(self, **kwargs):
        self.net = RBLNPytorchRuntime(self.model[0])
        self.cfg = cfg_re50
        self.cfg["pretrain"] = False
        self.batch_size = 1             # hard coded
        self.confidence_threshold = 0.7 # hard coded
        
    @classmethod
    def _get_rbln_config(
        cls,
        preprocessors: Union["AutoFeatureExtractor", "AutoProcessor", "AutoTokenizer"],
        model_config: "PretrainedConfig",
        rbln_kwargs: Dict[str, Any] = {},
    ) -> RBLNConfig:
        height = rbln_kwargs.get("height", 704)
        width = rbln_kwargs.get("width", 1280)
        input_info = [("frames", [1, 3, height, width], "float32")] # hard coded
        
        postprocessor_config = RBLNCompileConfig(
            compiled_model_name="postprocessor_0",
            input_info=input_info
        )
        
        rbln_config = RBLNConfig(
            rbln_cls=cls.__name__,
            compile_cfgs=[postprocessor_config],
            rbln_kwargs=rbln_kwargs,
        )
        return rbln_config
    
    @classmethod
    def get_compiled_model(cls, model, rbln_config: RBLNConfig):
        # RetinaFace
        postprocessors = model.net
        postprocessors_compiled_model = cls.compile(postprocessors, rbln_compile_config=rbln_config.compile_cfgs[0])

        return {"postprocessor_0" : postprocessors_compiled_model}
                
    @classmethod
    def _create_runtimes(
        cls,
        compiled_models: List[rebel.RBLNCompiledModel],
        rbln_device_map: Dict[str, int],
        activate_profiler: Optional[bool] = None,
    ) -> List[rebel.Runtime]:
        if "postprocessor_0" not in rbln_device_map:
            cls._raise_missing_compiled_file_error(["postprocessor_0"])

        device = rbln_device_map["postprocessor_0"]
        return [
            compiled_model.create_runtime(tensor_type="pt", device=device, activate_profiler=activate_profiler)
            for compiled_model in compiled_models
        ]
        
    def preprocess_frames(self, frames: np.ndarray) -> torch.Tensor:
        """Preprocess a sequence of frames for face detection.

        Args:
            frames: Input frames

        Returns:
            Preprocessed frames tensor
        """
        with torch.no_grad():
            frames_tensor = torch.from_numpy(frames).to(device=torch.device("cpu"), dtype=torch.float32)  # Shape: [T, H, W, C]
            frames_tensor = frames_tensor.permute(0, 3, 1, 2)  # Shape: [T, C, H, W]
            frames_tensor = frames_tensor[:, [2, 1, 0], :, :]  # RGB to BGR to match RetinaFace model input
            means = torch.tensor([104.0, 117.0, 123.0], device=torch.device("cpu"), dtype=torch.float32).view(1, 3, 1, 1)
            frames_tensor = frames_tensor - means  # Subtract mean BGR values for each channel
            return frames_tensor

    def postprocess(self, frames: np.ndarray) -> np.ndarray:
        """Blur faces in a sequence of frames.

        Args:
            frames: Input frames

        Returns:
            Processed frames with pixelated faces
        """
        # Create dataset and dataloader
        frames_tensor = self.preprocess_frames(frames)
        dataset = TensorDataset(frames_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        processed_frames, processed_batches = [], []

        prior_data, scale = None, None
        for i, batch in enumerate(dataloader):
            batch = batch[0]
            h, w = batch.shape[-2:]  # Batch shape: [C, H, W]

            with torch.no_grad():
                # Generate priors for the video
                if prior_data is None:
                    priorbox = PriorBox(self.cfg, image_size=(h, w))
                    priors = priorbox.forward()
                    priors = priors.to(device=torch.device("cpu"), dtype=torch.float32)
                    prior_data = priors.data

                # Get scale for resizing detections
                if scale is None:
                    scale = torch.Tensor([w, h, w, h])
                    scale = scale.to(device=torch.device("cpu"), dtype=torch.float32)

                batch_loc, batch_conf, _ = self.net(batch)

            # Blur detected faces in each batch of frames
            start_idx = i * self.batch_size
            end_idx = min(start_idx + self.batch_size, len(frames))
            processed_batches.append(
                self.blur_detected_faces(frames[start_idx:end_idx], batch_loc, batch_conf, prior_data, scale)
            )

        processed_frames = [frame for batch in processed_batches for frame in batch]
        return np.array(processed_frames)

class RBLNRuntimeSigLIPEncoder(RBLNPytorchRuntime):
    def __init__(self, runtime, **kwargs):
        super().__init__(runtime=runtime, **kwargs)
        self.model = runtime
        # self.processor = processor
        checkpoint_dir = COSMOS_GUARDRAIL_CHECKPOINT # snapshot_folder need
        import pathlib
        checkpoint_dir = (pathlib.Path(checkpoint_dir) / "video_content_safety_filter").as_posix()
        self.processor = SiglipProcessor.from_pretrained("google/siglip-so400m-patch14-384", cache_dir=checkpoint_dir)

    # @torch.inference_mode()
    def encode_image(self, input_img: PIL.Image.Image) -> torch.Tensor:
        """Encode an image into a feature vector."""
        with torch.no_grad():
            inputs = self.processor(images=input_img, return_tensors="pt").to(torch.device("cpu"), dtype=torch.float32)
            image_features = self.model(**inputs)
            image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features

class _SiglipVisionModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values):
        return self.model.get_image_features(
            pixel_values=pixel_values,
            return_dict=False,
        )

class RBLNAegis(Aegis): 
    @classmethod
    def from_model(
        cls, 
        model: torch.nn.Module, 
        rbln_config: Dict[str, Any] = {},
        model_save_dir: Optional[Union[str, Path, TemporaryDirectory]] = None,
        subfolder: str = "",
        **kwargs,):
        batch_size = rbln_config.get("batch_size", 1)
        tensor_parallel_size = rbln_config.get("tensor_parallel_size", 4)
        max_seq_len = rbln_config.get("max_seq_len", 4096)
        
        # aeigs = cls(model.tokenizer)
        llama = model.model.merge_and_unload()
        compiled_model = RBLNLlamaForCausalLM.from_model(
                    llama, 
                    export=True,
                    rbln_batch_size=batch_size,
                    rbln_max_seq_len=max_seq_len,
                    rbln_tensor_parallel_size=tensor_parallel_size,
                    model_save_dir=model_save_dir,
                    # subfolder=subfolder
                    )
        delattr(model, "model")
        setattr(model, "model", compiled_model)
        return model

    
    def filter_aegis_output(self, prompt: str) -> tuple[bool, str]:
        """Filter the Aegis model output and return the safety status and message."""
        full_prompt = self.get_moderation_prompt(prompt)
        inputs = self.tokenizer([full_prompt], add_special_tokens=False, return_tensors="pt").to(torch.device("cpu"))
        import pdb; pdb.set_trace()
        output = self.model.generate(**inputs, max_new_tokens=100, pad_token_id=self.tokenizer.eos_token_id)
        prompt_len = inputs["input_ids"].shape[-1]
        moderation_output = self.tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)

        if "unsafe" in moderation_output.lower():
            block_msg = self.get_aegis_block_message(moderation_output)
            return False, block_msg
        else:
            return True, ""

import importlib
class RBLNCosmosSafetyChecker(torch.nn.Module):
    original_class = CosmosSafetyChecker
    _submodules = {
        "text_guardrail": 
            {
                "safety_models" : [1]
                }, 
        "video_guardrail" : 
            {
                "postprocessors" : [0],
                "safety_models" : [0],
                },
        
        }
    # video_guardrail.safety_models[0].encoder.model
    @classmethod
    def _compile_submodules(cls, model, rbln_config, subfolder="", model_save_dir=""):
        save_dir_path = Path(model_save_dir)
        save_dir_path.mkdir(exist_ok=True)
        for guardrail_attr, targets in cls._submodules.items():
            for target, ids in targets.items():
                for id in ids:
                    target_list = getattr(getattr(model, guardrail_attr), target)
                    compile_target = target_list.pop(id) # aegis
                    rbln_class = globals()[f"RBLN{compile_target.__class__.__name__}"]
                    
                    compiled_model = rbln_class.from_model(compile_target,
                                                           export=True, 
                                                           model_save_dir=f'{model_save_dir}/{guardrail_attr}',
                                                           subfolder=f"{target}",
                                                           rbln_config=rbln_config)
                    del compile_target
                    target_list.insert(id, compiled_model)
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