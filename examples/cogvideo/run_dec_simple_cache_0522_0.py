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
from transformers import PretrainedConfig

from optimum.rbln.modeling import RBLNModel
from optimum.rbln.modeling_config import DEFAULT_COMPILED_MODEL_NAME, RBLNCompileConfig, RBLNConfig
from optimum.rbln.utils.logging import get_logger
from optimum.rbln.diffusers.models.autoencoders.vae_test import (
    RBLNRuntimeVAECogVideoXDecoder,
    RBLNRuntimeVAECogVideoXEncoder,
    _VAECogVideoXEncoder,
)

# MODE = "flatten" # "reshape" # "origin"
MODE = "reshape" # "origin"
# MODE = "origin"

if TYPE_CHECKING:
    import torch
    from transformers import AutoFeatureExtractor, AutoProcessor, AutoTokenizer, PretrainedConfig

logger = get_logger(__name__)

CONV_CACHE = {
    "conv_in": (1, 16, 2, 60, 90),
    "mid_block.resnet_0.conv1": (1, 512, 2, 60, 90),
    "mid_block.resnet_0.conv2": (1, 512, 2, 60, 90),
    "mid_block.resnet_1.conv1": (1, 512, 2, 60, 90),
    "mid_block.resnet_1.conv2": (1, 512, 2, 60, 90),
    "up_block_0.resnet_0.conv1": (1, 512, 2, 60, 90),
    "up_block_0.resnet_0.conv2": (1, 512, 2, 60, 90),
    "up_block_0.resnet_1.conv1": (1, 512, 2, 60, 90),
    "up_block_0.resnet_1.conv2": (1, 512, 2, 60, 90),
    "up_block_0.resnet_2.conv1": (1, 512, 2, 60, 90),
    "up_block_0.resnet_2.conv2": (1, 512, 2, 60, 90),
    "up_block_0.resnet_3.conv1": (1, 512, 2, 60, 90),
    "up_block_0.resnet_3.conv2": (1, 512, 2, 60, 90),
    "up_block_1.resnet_0.conv1": (1, 512, 2, 120, 180),
    "up_block_1.resnet_0.conv2": (1, 256, 2, 120, 180),
    "up_block_1.resnet_1.conv1": (1, 256, 2, 120, 180),
    "up_block_1.resnet_1.conv2": (1, 256, 2, 120, 180),
    "up_block_1.resnet_2.conv1": (1, 256, 2, 120, 180),
    "up_block_1.resnet_2.conv2": (1, 256, 2, 120, 180),
    "up_block_1.resnet_3.conv1": (1, 256, 2, 120, 180),
    "up_block_1.resnet_3.conv2": (1, 256, 2, 120, 180),
    "up_block_2.resnet_0.conv1": (1, 256, 2, 240, 360),
    "up_block_2.resnet_0.conv2": (1, 256, 2, 240, 360),
    "up_block_2.resnet_1.conv1": (1, 256, 2, 240, 360),
    "up_block_2.resnet_1.conv2": (1, 256, 2, 240, 360),
    "up_block_2.resnet_2.conv1": (1, 256, 2, 240, 360),
    "up_block_2.resnet_2.conv2": (1, 256, 2, 240, 360),
    "up_block_2.resnet_3.conv1": (1, 256, 2, 240, 360),
    "up_block_2.resnet_3.conv2": (1, 256, 2, 240, 360),
    "up_block_3.resnet_0.conv1": (1, 256, 2, 480, 720),
    "up_block_3.resnet_0.conv2": (1, 128, 2, 480, 720),
    "up_block_3.resnet_1.conv1": (1, 128, 2, 480, 720),
    "up_block_3.resnet_1.conv2": (1, 128, 2, 480, 720),
    "up_block_3.resnet_2.conv1": (1, 128, 2, 480, 720),
    "up_block_3.resnet_2.conv2": (1, 128, 2, 480, 720),
    "up_block_3.resnet_3.conv1": (1, 128, 2, 480, 720),
    "up_block_3.resnet_3.conv2": (1, 128, 2, 480, 720),
    "conv_out": (1, 128, 2, 480, 720)
}

from optimum.rbln.ops import register_rbln_custom_cache_update

class _VAECogVideoXDecoder(torch.nn.Module):
    def __init__(self, cog_video_x: AutoencoderKLCogVideoX):
        super().__init__()
        register_rbln_custom_cache_update()
        self.cog_video_x = cog_video_x
        # self.keys = None
        self.keys = self.get_conv_cache_key()
        self.conv_out = torch.nn.Conv3d(256,3,3,padding=1,bias=False)

    def _to_tuple(self, conv_cache):
        conv_cache_list = []
        keys = []
        def isdict(obj, names):
            if isinstance(obj, dict):
                for _k, _v in obj.items():
                    isdict(_v, names+f".{_k}")
            else :
                if "norm" not in names:
                # if "norm" not in names and "up_block_2" not in names and "up_block_3" not in names and "conv_out" not in names:
                    conv_cache_list.append(obj)
                    keys.append(names)
                # if names in set(self.keys):
                #     conv_cache_list.append(obj)
        
        for k, v in conv_cache.items():
            isdict(v, k)
        return tuple(conv_cache_list), keys

    def _to_nested_dict(self, conv_cache_list, keys):
        conv_cache_dict = {}
        for k, v in zip(keys, conv_cache_list):
            parts = k.split(".")
            current = conv_cache_dict
            
            # 마지막 부분을 제외한 모든 부분에 대해 중첩 딕셔너리 생성
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            
            # 마지막 부분에 값 할당
            current[parts[-1]] = v
        return conv_cache_dict

    def get_conv_cache_key(self):
        keys = []
        for k in CONV_CACHE.keys():
            if "norm" not in k and k in CONV_CACHE_mod:
                keys.append(k)
        return keys
        
    def forward(self, z: torch.Tensor, *args: Optional[Tuple[torch.Tensor]]):
        enc = z.shape[2] > 2 # FIXME make generalized condition
        dummy_outs=[]

        if enc :
            cov_video_dec_out, conv_cache = self.cog_video_x.decoder(z)
            # # temp
            # cov_video_dec_out = self.conv_out(cov_video_dec_out)
            conv_cache_list, _ = self._to_tuple(conv_cache)
            
            for cache, _conv_cache in zip(args, conv_cache_list):
                # cache.shape : n, c, d, h, w
                # _conv_cache.shape : n, c, d, h, w
                
                batch_dim = torch.tensor(0, dtype=torch.int16)
                batch_axis = torch.tensor(0, dtype=torch.int16)

                if MODE == "flatten":
                    # flattened
                    conv_cache_reshaped = _conv_cache.reshape(_conv_cache.shape[0], -1) # (b, cdhw)

                elif MODE == "reshape":
                    # reshaped
                    conv_cache_reshaped = _conv_cache.permute(0,2,3,4,1) # (b, cdhw)
                
                elif MODE == "origin":
                    # origin
                    conv_cache_reshaped = _conv_cache
                
                # import pdb; pdb.set_trace()
                assert (cache.shape == conv_cache_reshaped.shape), print(cache.shape, conv_cache_reshaped.shape)
                dummy_out = torch.ops.rbln_custom_ops.rbln_cache_update(cache, conv_cache_reshaped, batch_dim, batch_axis)
                dummy_outs.append(dummy_out)
            
            return cov_video_dec_out, torch.stack(tuple(dummy_outs))
                
        else :
            conv_cache_dict = self._to_nested_dict(args, self.keys)
            
            cov_video_dec_out, conv_cache = self.cog_video_x.decoder(z, conv_cache=conv_cache_dict)
            # # temp
            # cov_video_dec_out = self.conv_out(cov_video_dec_out)
            conv_cache_list, _ = self._to_tuple(conv_cache)
            
            for cache, _conv_cache in zip(args, conv_cache_list):
                batch_dim = torch.tensor(0, dtype=torch.int16)
                batch_axis = torch.tensor(0, dtype=torch.int16)
                
                if MODE == "flatten":
                    # flattened
                    conv_cache_reshaped = _conv_cache.reshape(_conv_cache.shape[0], -1) # (b, cdhw)

                elif MODE == "reshape":
                    # reshaped
                    conv_cache_reshaped = _conv_cache.permute(0,2,3,4,1) # (b, d, h, w, c)
                
                elif MODE == "origin":
                    # origin
                    conv_cache_reshaped = _conv_cache
                
                # import pdb; pdb.set_trace()
                assert (cache.shape == conv_cache_reshaped.shape)
                dummy_out = torch.ops.rbln_custom_ops.rbln_cache_update(cache, conv_cache_reshaped, batch_dim, batch_axis)
                dummy_outs.append(dummy_out)
            
        # return cov_video_dec_out, conv_cache_list, torch.stack(tuple(dummy_outs))
        # return cov_video_dec_out
        
        return cov_video_dec_out, torch.stack(tuple(dummy_outs))

class RBLNAutoencoderKLCogVideoX(RBLNModel):
    auto_model_class = AutoencoderKLCogVideoX
    config_name = "config.json"
    hf_library_name = "diffusers"

    def __post_init__(self, **kwargs):
        super().__post_init__(**kwargs)

        if self.rbln_config.model_cfg.get("img2vid_pipeline"):
            self.encoder = RBLNRuntimeVAECogVideoXEncoder(runtime=self.model[0], main_input_name="x")
            self.decoder = RBLNRuntimeVAECogVideoXDecoder(runtime=self.model[1], main_input_name="z")
        else:
            self.decoder = RBLNRuntimeVAECogVideoXDecoder(runtime=self.model[0], main_input_name="z")

        # self.image_size = self.rbln_config.model_cfg["sample_size"]

    @classmethod
    def get_compiled_model(cls, model, rbln_config: RBLNConfig):
        def compile_img2vid():
            encoder_model = _VAECogVideoXEncoder(model)
            decoder_model = _VAECogVideoXDecoder(model)
            encoder_model.eval()
            decoder_model.eval()

            enc_compiled_model = cls.compile(encoder_model, rbln_compile_config=rbln_config.compile_cfgs[0])
            dec_compiled_model = cls.compile(decoder_model, rbln_compile_config=rbln_config.compile_cfgs[1])

            return {"encoder": enc_compiled_model, "decoder": dec_compiled_model}

        def compile_txt2vid():
            def rbln_decorator(func):
                def wrapper(*args, **kwargs):                    
                    b, c, d, h, w = args[0].shape # z.shape
                    conv_cache = kwargs.get("conv_cache") # None
                    if conv_cache is not None :
                        if len(conv_cache) < 3:
                            conv_cache = conv_cache.view(b, c, 2, h, w)
                    elif len(args) > 1 :
                        if args[-1] is not None and len(args[-1]) < 3:
                            conv_cache = args[-1].view(b, c, 2, h, w)
                            # import pdb; pdb.set_trace()
                    
                    result = func(args[0], conv_cache=conv_cache)
                    return result
                return wrapper
            
            decoder_model_0 = _VAECogVideoXDecoder(model)
            
            # for rbln cache update
            for m in model.modules():
                from diffusers.models.autoencoders.autoencoder_kl_cogvideox import CogVideoXCausalConv3d
                if isinstance(m, CogVideoXCausalConv3d):
                    m.fake_context_parallel_forward = rbln_decorator(m.fake_context_parallel_forward)
                    
            decoder_model_1 = _VAECogVideoXDecoder(model)

            from rebel.compile_context import CompileContext
            context = CompileContext(use_weight_sharing=False)

            enc_example_inputs = rbln_config.compile_cfgs[0].get_dummy_inputs(fill=0) # is it needed?

            # # Mark encoder's static tensors (cross kv states) FIXME decoder_0 output conv cache
            static_tensors = {}
            # import pdb; pdb.set_trace()
            for (name, _, _), tensor in zip(rbln_config.compile_cfgs[0].input_info, enc_example_inputs):
                if "conv" in name:
                    static_tensors[name] = tensor
                    context.mark_static_address(tensor)

            dec_example_inputs = rbln_config.compile_cfgs[1].get_dummy_inputs(fill=0, static_tensors=static_tensors)

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

            return (dec_compiled_model_0, dec_compiled_model_1)

        if rbln_config.model_cfg.get("img2vid_pipeline"):
            return compile_img2vid()
        else:
            return compile_txt2vid()

    @classmethod
    def get_vae_sample_size(cls, pipe, rbln_config: Dict[str, Any]) -> Union[int, Tuple[int, int]]:
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
    def update_rbln_config_using_pipe(cls, pipe, rbln_config: Dict[str, Any]) -> Dict[str, Any]:
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
        rbln_batch_size = rbln_kwargs.get("batch_size", 1)
        sample_size = rbln_kwargs.get("sample_size", (480, 720))
        is_img2vid = rbln_kwargs.get("is_img2vid", False)
        num_frames = rbln_kwargs.get("num_frames", 49)

        if rbln_batch_size is None:
            rbln_batch_size = 1

        vae_scale_factor_spatial = rbln_kwargs.get("vae_scale_factor_spatial", 8)
        vae_scale_factor_temporal = rbln_kwargs.get("vae_scale_factor_temporal", 4)
        num_latent_frames_batch_size = rbln_kwargs.get("num_latent_frames_batch_size", 2)

        dec_shape = (sample_size[0] // vae_scale_factor_spatial, sample_size[1] // vae_scale_factor_spatial)

        if is_img2vid:
            pass
        
        vae_dec_input_info_0=[
                (
                    "z",
                    [
                        rbln_batch_size,
                        model_config.latent_channels,
                        num_latent_frames_batch_size + num_frames%num_latent_frames_batch_size,
                        # num_latent_frames_batch_size,
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
            if "norm" not in k and k in CONV_CACHE_mod:
                n, c, d, h, w = v
                
                if MODE == "flatten":
                    # flattened
                    _input_info_1 = (k, [n, c*d*h*w],  "float32")
                
                elif MODE == "reshape":
                    # Reshaped
                    _input_info_1 = (k, [n, d, h, w, c],  "float32")
                
                elif MODE == "origin":
                    # origin
                    _input_info_1 = (k, [n, c, d, h, w],  "float32")
                else :
                    raise NotImplementedError("Select MODE")
                
                vae_dec_input_info_0.extend([_input_info_1])
                vae_dec_input_info_1.extend([_input_info_1])
                
        
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
            if DEFAULT_COMPILED_MODEL_NAME not in rbln_device_map:
                cls._raise_missing_compiled_file_error([DEFAULT_COMPILED_MODEL_NAME])

            device_val = rbln_device_map[DEFAULT_COMPILED_MODEL_NAME]
            return [
                compiled_models[0].create_runtime(
                    tensor_type="pt", device=device_val, activate_profiler=activate_profiler
                )
            ]

        if any(model_name not in rbln_device_map for model_name in ["encoder", "decoder"]):
            cls._raise_missing_compiled_file_error(["encoder", "decoder"])

        device_vals = [rbln_device_map["encoder"], rbln_device_map["decoder"]]
        return [
            compiled_model.create_runtime(tensor_type="pt", device=device_val, activate_profiler=activate_profiler)
            for compiled_model, device_val in zip(compiled_models, device_vals)
        ]

    def encode(self, x: torch.FloatTensor, **kwargs) -> torch.FloatTensor:
        posterior = self.encoder.encode(x)
        return AutoencoderKLOutput(latent_dist=posterior)

    def decode(self, z: torch.FloatTensor, **kwargs) -> torch.FloatTensor:
        return self.decoder.decode(z)
        

if __name__ == "__main__":
    model_id = "THUDM/CogVideoX-2b"
    
    config = AutoencoderKLCogVideoX.load_config(model_id, subfolder="vae")
    config['layers_per_block']=1
    
    # model = AutoencoderKLCogVideoX.from_pretrained(model_id, subfolder="vae")
    model = AutoencoderKLCogVideoX.from_config(config)
    # import pdb; pdb.set_trace()
    # from diffusers import CogVideoXPipeline
    # pipe = CogVideoXPipeline.from_pretrained(pretrained_model_name_or_path=model_id)
    # model = pipe.vae
    
    class identity_model(torch.nn.Module):
        def __init__(self,):
            super().__init__()
        
        def forward(self, x, *args, **kwargs):
            return x, torch.randn(256)
        
    # model.decoder.up_blocks = model.decoder.up_blocks[:1]
    # model.decoder.up_blocks = model.decoder.up_blocks[:2]
    # model.decoder.norm_out = identity_model()
    # model.decoder.conv_act = identity_model()
    # model.decoder.conv_out = identity_model()
    
    input = torch.randn(1,16,3,60,90)
    
    output, cache = model.decoder(input)
    _, CONV_CACHE_mod = _VAECogVideoXDecoder._to_tuple(model, cache)
    
    rbln_model = RBLNAutoencoderKLCogVideoX.from_model(
        model=model,
        export=True,
        model_save_dir="rbln_cog_vae2",
    )
    with torch.no_grad():
        outputs = model.decoder(input)
        output_pt, ca_pt = outputs[0][0], outputs[1]
        
        outputs = rbln_model.decoder(input)
        output, ca = outputs[0], outputs[1:]

        from scipy.stats import pearsonr
        p = pearsonr(output_pt.detach().flatten().numpy(), output.detach().flatten().numpy())[0]
        print(p)
    