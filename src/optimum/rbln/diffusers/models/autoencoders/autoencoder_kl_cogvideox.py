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

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import rebel
import torch  # noqa: I001
from diffusers import AutoencoderKLCogVideoX
from diffusers.models.modeling_outputs import AutoencoderKLOutput
from diffusers.models.autoencoders.vae import DecoderOutput
from transformers import PretrainedConfig

from ....modeling import RBLNModel
from ....modeling_config import DEFAULT_COMPILED_MODEL_NAME, RBLNCompileConfig, RBLNConfig
from ....utils.logging import get_logger
from ...modeling_diffusers import RBLNDiffusionMixin
from .vae import (
    RBLNRuntimeVAECogVideoXDecoder,
    RBLNRuntimeVAECogVideoXEncoder,
    _VAECogVideoXDecoder,
    _VAECogVideoXEncoder,
)


if TYPE_CHECKING:
    import torch
    from transformers import AutoFeatureExtractor, AutoProcessor, AutoTokenizer, PretrainedConfig

logger = get_logger(__name__)

from .input_dict import conv_cache as CONV_CACHE

class RBLNAutoencoderKLCogVideoX(RBLNModel):
    auto_model_class = AutoencoderKLCogVideoX
    config_name = "config.json"
    hf_library_name = "diffusers"

    def __post_init__(self, **kwargs):
        super().__post_init__(**kwargs)
        if self.rbln_config.model_cfg.get("img2vid_pipeline"):
            self.encoder = RBLNRuntimeVAECogVideoXEncoder(runtime=self.model[0], main_input_name="x")
            self.decoder_0 = RBLNRuntimeVAECogVideoXDecoder(runtime=self.model[1], main_input_name="z")
            self.decoder_1 = RBLNRuntimeVAECogVideoXDecoder(runtime=self.model[2], main_input_name="z")
        else:
            self.decoder_0 = RBLNRuntimeVAECogVideoXDecoder(runtime=self.model[0], main_input_name="z")
            self.decoder_1 = RBLNRuntimeVAECogVideoXDecoder(runtime=self.model[1], main_input_name="z")

        self.image_size = self.rbln_config.model_cfg["sample_size"]
        self.num_latent_frames_batch_size = self.rbln_config.model_cfg["num_latent_frames_batch_size"]
        

    @classmethod
    def get_compiled_model(cls, model, rbln_config: RBLNConfig):
        def compile_img2vid():
            encoder_model = _VAECogVideoXEncoder(model)
            decoder_model_0 = _VAECogVideoXDecoder(model)
            decoder_model_1 = _VAECogVideoXDecoder(model)
            
            encoder_model.eval()
            decoder_model_0.eval()
            decoder_model_1.eval()

            enc_compiled_model = cls.compile(encoder_model, rbln_compile_config=rbln_config.compile_cfgs[0])
            
            dec_compiled_model_0 = cls.compile(decoder_model_0, rbln_compile_config=rbln_config.compile_cfgs[1])
            dec_compiled_model_1 = cls.compile(decoder_model_1, rbln_compile_config=rbln_config.compile_cfgs[2])

            return {"encoder": enc_compiled_model, "decoder": (dec_compiled_model_0, dec_compiled_model_1)}

        def compile_txt2vid():
            def rbln_decorator(func):
                def wrapper(*args, **kwargs):
                    # b, c, d, h, w = args[0].shape # z
                    conv_cache = kwargs.get("conv_cache") # None
                    if conv_cache is not None :
                        if len(conv_cache) < 3:
                            # conv_cache = conv_cache.view(b, c, 2, h, w)
                            conv_cache = conv_cache.permute(0,4,1,2,3)
                    elif len(args) > 1 :
                        if args[-1] is not None and len(args[-1]) < 3:
                            # conv_cache = args[-1].view(b, c, 2, h, w)
                            conv_cache = args[-1].permute(0,4,1,2,3)
                    
                    result = func(args[0], conv_cache=conv_cache) # todo(si) 일반화? 고도화? functionality 검증?
                    return result
                return wrapper
            
            # for rbln cache update
            for m in model.modules():
                from diffusers.models.autoencoders.autoencoder_kl_cogvideox import CogVideoXCausalConv3d
                if isinstance(m, CogVideoXCausalConv3d):
                    m.fake_context_parallel_forward = rbln_decorator(m.fake_context_parallel_forward)
            decoder_model_0 = _VAECogVideoXDecoder(model)        
            decoder_model_1 = _VAECogVideoXDecoder(model)
            # decoder_model_0 = _VAECogVideoXDecoder(model)
            # decoder_model_1 = _VAECogVideoXDecoder(model)

            from rebel.compile_context import CompileContext
            context = CompileContext(use_weight_sharing=False) # NOTE(si) : True possible?

            enc_example_inputs = rbln_config.compile_cfgs[0].get_dummy_inputs(fill=0) # is it needed?

            # # Mark encoder's static tensors (cross kv states) FIXME decoder_0 output conv cache
            static_tensors = {}
            for (name, _, _), tensor in zip(rbln_config.compile_cfgs[0].input_info, enc_example_inputs):
                if "conv" in name:
                    static_tensors[name] = tensor
                    context.mark_static_address(tensor)

            dec_example_inputs = rbln_config.compile_cfgs[1].get_dummy_inputs(fill=0, static_tensors=static_tensors)

            # Mark decoder's static tensors (self kv states)
            for (name, _, _), tensor in zip(rbln_config.compile_cfgs[1].input_info, dec_example_inputs):
                if "conv" in name:
                    context.mark_static_address(tensor)

            dec_compiled_model_0 = cls.compile(
                decoder_model_0.eval(),
                rbln_config.compile_cfgs[0],
                example_inputs=enc_example_inputs,
                compile_context=context,
            )

            dec_compiled_model_1 = cls.compile(
                decoder_model_1.eval(),
                rbln_config.compile_cfgs[1],
                example_inputs=dec_example_inputs,
                compile_context=context,
            )

            ###############################################

            # decoder_model_0.eval()
            # decoder_model_1.eval()
            
            # dec_compiled_model_0 = cls.compile(decoder_model_0, rbln_compile_config=rbln_config.compile_cfgs[0])
            # dec_compiled_model_1 = cls.compile(decoder_model_1, rbln_compile_config=rbln_config.compile_cfgs[1])

            return (dec_compiled_model_0, dec_compiled_model_1)

        if rbln_config.model_cfg.get("img2vid_pipeline"):
            return compile_img2vid()
        else:
            return compile_txt2vid()

    @classmethod
    def get_vae_sample_size(cls, pipe: RBLNDiffusionMixin, rbln_config: Dict[str, Any]) -> Union[int, Tuple[int, int]]:
        image_size = (rbln_config.get("img_height"), rbln_config.get("img_width"))

        noise_module = getattr(pipe, "unet", None) or getattr(pipe, "transformer", None)
        vae_scale_factor = (
            pipe.vae_scale_factor
            if hasattr(pipe, "vae_scale_factor")
            else 2 ** (len(pipe.vae.config.block_out_channels) - 1)
        )

        if noise_module is None:
            raise AttributeError(
                "Cannot find noise processing or predicting module attributes. ex. U-Net, Transformer, ..."
            )

        if (image_size[0] is None) != (image_size[1] is None):
            raise ValueError("Both image height and image width must be given or not given")

        elif image_size[0] is None and image_size[1] is None:
            noise_module_sample_size = (noise_module.config.sample_height, noise_module.config.sample_width)
            sample_size = (
                noise_module_sample_size[0] * vae_scale_factor,
                noise_module_sample_size[1] * vae_scale_factor,
            )
        else:
            sample_size = (image_size[0], image_size[1])

        return sample_size

    @classmethod
    def update_rbln_config_using_pipe(cls, pipe: RBLNDiffusionMixin, rbln_config: Dict[str, Any]) -> Dict[str, Any]:
        num_frames = rbln_config.get("num_frames")
        if num_frames is None:
            num_frames = pipe.transformer.config.sample_frames

        rbln_config.update(
            {
                "num_frames": num_frames,
                "sample_size": cls.get_vae_sample_size(pipe, rbln_config),
                "vae_scale_factor_spatial": pipe.vae_scale_factor_spatial,
                "vae_scale_factor_temporal": pipe.vae_scale_factor_temporal,
                "num_latent_frames_batch_size": pipe.vae.num_latent_frames_batch_size
            }
        )
        return rbln_config

    @classmethod
    def _get_rbln_config(
        cls,
        preprocessors: Union["AutoFeatureExtractor", "AutoProcessor", "AutoTokenizer"],
        model_config: "PretrainedConfig",
        rbln_kwargs: Dict[str, Any] = {},
    ) -> RBLNConfig:
        rbln_batch_size = rbln_kwargs.get("batch_size")
        sample_size = rbln_kwargs.get("sample_size")
        is_img2vid = rbln_kwargs.get("is_img2vid")
        num_frames = rbln_kwargs.get("num_frames")

        if rbln_batch_size is None:
            rbln_batch_size = 1

        vae_scale_factor_spatial = rbln_kwargs["vae_scale_factor_spatial"]
        vae_scale_factor_temporal = rbln_kwargs["vae_scale_factor_temporal"]
        num_latent_frames_batch_size = rbln_kwargs.get("num_latent_frames_batch_size")

        dec_shape = (sample_size[0] // vae_scale_factor_spatial, sample_size[1] // vae_scale_factor_spatial)

        if is_img2vid:
            enc_shape = (sample_size[0], sample_size[1])
            vae_enc_input_info = [
                (
                    "x",
                    [
                        rbln_batch_size,
                        model_config.in_channels,
                        (num_frames - 1) // vae_scale_factor_temporal + 1,
                        enc_shape[0],
                        enc_shape[1],
                    ],
                    "float32",
                )
            ]
            
            vae_dec_input_info_0=[
                    (
                        "z",
                        [
                            rbln_batch_size,
                            model_config.latent_channels,
                            num_latent_frames_batch_size + num_frames%num_latent_frames_batch_size,
                            dec_shape[0],
                            dec_shape[1],
                        ],
                        "float32",
                    )
                ]
            vae_dec_input_info_1=[
                    (
                        "z",
                        [
                            rbln_batch_size,
                            model_config.latent_channels,
                            num_latent_frames_batch_size,
                            dec_shape[0], 
                            dec_shape[1],
                        ],
                        "float32",
                    )
                ]

            for k, v in CONV_CACHE.items():
                if "norm" not in k :
                    _input_info = (k, list(v), "float32")
                    vae_dec_input_info_0.extend([_input_info])
                    vae_dec_input_info_1.extend([_input_info])

            enc_rbln_compile_config = RBLNCompileConfig(compiled_model_name="encoder", input_info=vae_enc_input_info)
            dec_0_rbln_compile_config = RBLNCompileConfig(compiled_model_name="decoder_0", input_info=vae_dec_input_info_0)
            dec_1_rbln_compile_config = RBLNCompileConfig(compiled_model_name="decoder_1", input_info=vae_dec_input_info_1)

            compile_cfgs = [enc_rbln_compile_config, dec_0_rbln_compile_config, dec_1_rbln_compile_config]
            rbln_config = RBLNConfig(
                rbln_cls=cls.__name__,
                compile_cfgs=compile_cfgs,
                rbln_kwargs=rbln_kwargs,
            )
            return rbln_config

        vae_dec_input_info_0=[
                (
                    "z",
                    [
                        rbln_batch_size,
                        model_config.latent_channels,
                        num_latent_frames_batch_size + num_frames%num_latent_frames_batch_size,
                        dec_shape[0],
                        dec_shape[1],
                    ],
                    "float32",
                )
            ]
        vae_dec_input_info_1=[
                (
                    "z",
                    [
                        rbln_batch_size,
                        model_config.latent_channels,
                        num_latent_frames_batch_size,
                        dec_shape[0], 
                        dec_shape[1],
                    ],
                    "float32",
                )
            ]

        for k, v in CONV_CACHE.items():
            if "norm" not in k :
                n,c,d,h,w = v
                v = (n,d,h,w,c)
                _input_info = (k, list(v), "float32")
                vae_dec_input_info_0.extend([_input_info])
                vae_dec_input_info_1.extend([_input_info])

        dec_0_rbln_compile_config = RBLNCompileConfig(compiled_model_name="decoder_0", input_info=vae_dec_input_info_0)
        dec_1_rbln_compile_config = RBLNCompileConfig(compiled_model_name="decoder_1", input_info=vae_dec_input_info_1)

        rbln_config = RBLNConfig(
            rbln_cls=cls.__name__,
            compile_cfgs=[dec_0_rbln_compile_config, dec_1_rbln_compile_config],
            rbln_kwargs=rbln_kwargs,
        )
        return rbln_config

    @classmethod
    def _create_runtimes(
        cls,
        compiled_models: List[rebel.RBLNCompiledModel],
        rbln_device_map: Dict[str, int],
        activate_profiler: Optional[bool] = None,
    ) -> List[rebel.Runtime]:
        if len(compiled_models) == 1:
            raise ValueError("compiled_models argument has two models")
            if DEFAULT_COMPILED_MODEL_NAME not in rbln_device_map:
                cls._raise_missing_compiled_file_error([DEFAULT_COMPILED_MODEL_NAME])

            device_val = rbln_device_map[DEFAULT_COMPILED_MODEL_NAME]
            return [
                compiled_models[0].create_runtime(
                    tensor_type="pt", device=device_val, activate_profiler=activate_profiler
                )
            ]
        elif len(compiled_models) == 2: # TODO(si) WIP
            if any(model_name not in rbln_device_map for model_name in ["decoder_0", "decoder_1"]):
                cls._raise_missing_compiled_file_error(["decoder_0", "decoder_1"])

            device_vals = [rbln_device_map["decoder_0"], rbln_device_map["decoder_1"]]
            return [
                compiled_model.create_runtime(tensor_type="pt", device=device_val, activate_profiler=activate_profiler)
                for compiled_model, device_val in zip(compiled_models, device_vals)
            ]

        if any(model_name not in rbln_device_map for model_name in ["encoder", "decoder_0", "decoder_1"]):
            cls._raise_missing_compiled_file_error(["encoder", "decoder_0", "decoder_1"])

        device_vals = [rbln_device_map["encoder"], rbln_device_map["decoder_0"], rbln_device_map["decoder_1"]]
        return [
            compiled_model.create_runtime(tensor_type="pt", device=device_val, activate_profiler=activate_profiler)
            for compiled_model, device_val in zip(compiled_models, device_vals)
        ]

    def encode(self, x: torch.FloatTensor, **kwargs) -> torch.FloatTensor:
        posterior = self.encoder.encode(x)
        return AutoencoderKLOutput(latent_dist=posterior)

    def _decode(self, z: torch.Tensor, return_dict: bool = True) -> Union[DecoderOutput, torch.Tensor]:
        batch_size, num_channels, num_frames, height, width = z.shape

        # if self.cog_video_x.use_tiling and (
        #     width > self.cog_video_x.tile_latent_min_width or height > self.cog_video_x.tile_latent_min_height
        # ):
        #     raise ValueError("Optimum-RBLN doesn't support tiled decoding aross H,W axis")

        frame_batch_size = self.num_latent_frames_batch_size
        num_batches = max(num_frames // frame_batch_size, 1)
        # conv_cache = None
        dec = []
        remaining_frames = num_frames % frame_batch_size
        outputs = self.decoder_0(z[:, :, :frame_batch_size+remaining_frames])
        z_intermediate = outputs[0]
        import pdb; pdb.set_trace()
        dec.append(z_intermediate)

        for i in range(1, num_batches):
            # remaining_frames = num_frames % frame_batch_size
            start_frame = frame_batch_size * i + remaining_frames
            end_frame = frame_batch_size * (i + 1) + remaining_frames
            z_intermediate = z[:, :, start_frame:end_frame]
            # if self.cog_video_x.post_quant_conv is not None:
            #     z_intermediate = self.cog_video_x.post_quant_conv(z_intermediate)
            # import pdb; pdb.set_trace()
            
            # z_intermediate, conv_cache = self.decoder_1(z_intermediate, conv_cache=conv_cache)
            outputs = self.decoder_1(z_intermediate)
            import pdb; pdb.set_trace()
            z_intermediate = outputs[0]
            # z_intermediate, conv_cache, dummy = outputs[0], outputs[1:1+len(outputs)//2], outputs[1+len(outputs)//2]
            # z_intermediate, conv_cache = output[0], output[1:]
            # outputs, _ = self.decoder_1(z_intermediate)
            dec.append(z_intermediate)

        dec = torch.cat(dec, dim=2)

        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)
        
    
    def decode(self, z: torch.FloatTensor, return_dict: bool = True) -> torch.FloatTensor:
        decoded = self._decode(z).sample

        if not return_dict:
            return (decoded,)
        return DecoderOutput(sample=decoded)
