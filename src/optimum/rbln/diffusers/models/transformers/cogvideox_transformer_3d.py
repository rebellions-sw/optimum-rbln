import logging
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union

import torch
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.transformers.cogvideox_transformer_3d import CogVideoXTransformer3DModel
from transformers import PretrainedConfig

from ....modeling import RBLNModel
from ....modeling_config import RBLNCompileConfig, RBLNConfig
from ...modeling_diffusers import RBLNDiffusionMixin


if TYPE_CHECKING:
    from transformers import AutoFeatureExtractor, AutoProcessor, AutoTokenizer

logger = logging.getLogger(__name__)


class CogVideoXTransformer3DModelWrapper(torch.nn.Module):
    def __init__(self, model: "CogVideoXTransformer3DModel") -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: Union[int, float, torch.LongTensor],
        timestep_cond: Optional[torch.Tensor] = None,
        ofs: Optional[Union[int, float, torch.LongTensor]] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ):
        return self.model(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            return_dict=False,
        )


class RBLNCogVideoXTransformer3DModel(RBLNModel):
    hf_library_name = "diffusers"
    
    def __post_init__(self, **kwargs):
        super().__post_init__(**kwargs)

    @classmethod
    def wrap_model_if_needed(cls, model: torch.nn.Module, rbln_config: RBLNConfig) -> torch.nn.Module:
        return CogVideoXTransformer3DModelWrapper(model).eval()

    @classmethod
    def update_rbln_config_using_pipe(cls, pipe: RBLNDiffusionMixin, rbln_config: Dict[str, Any]) -> Dict[str, Any]:
        sample_size = rbln_config.get("sample_size", None)
        vae_scale_factor_spatial = pipe.vae_scale_factor_spatial

        num_frames = rbln_config.get("num_frames")
        img_width = rbln_config.get("img_width", None)
        img_height = rbln_config.get("img_height", None)

        if (img_height is None) ^ (img_width is None):
            raise ValueError("Both image height and image width must be given or not given")

        elif img_height and img_width:
            sample_size = img_height // vae_scale_factor_spatial, img_width // vae_scale_factor_spatial

        batch_size = rbln_config.get("batch_size")
        if not batch_size:
            do_classifier_free_guidance = rbln_config.get("guidance_scale", 5.0) > 1.0
            batch_size = 2 if do_classifier_free_guidance else 1
        else:
            if rbln_config.get("guidance_scale"):
                logger.warning(
                    "guidance_scale is ignored because batch size is explicitly specified. "
                    "To ensure consistent behavior, consider removing the guidance scale or "
                    "adjusting the batch size configuration as needed."
                )

        rbln_config.update(
            {
                "batch_size": batch_size,
                "sample_size": sample_size,
                "num_frames": num_frames,
                "vae_scale_factor_temporal": pipe.vae_scale_factor_temporal,
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
        rbln_batch_size = rbln_kwargs.get("batch_size", None)
        sample_size = rbln_kwargs.get("sample_size")
        num_frames = rbln_kwargs.get("num_frames")

        if sample_size is None:
            # NOTE(si): From diffusers >= v0.32.0, pipe.transformer.config.sample_height and pipe.transformer.config.sample_width is used explicitly.
            sample_size = model_config.sample_height, model_config.sample_width
            rbln_kwargs["sample_size"] = sample_size
        
        if num_frames is None:
            num_frames = model_config.sample_frames
        
        vae_scale_factor_temporal = rbln_kwargs.get("vae_scale_factor_temporal", None)

        rbln_max_seqeunce_length = rbln_kwargs.get("max_sequence_length")
        if rbln_max_seqeunce_length is None:
            rbln_max_sequence_length = model_config.max_text_seq_length

        input_info = [
            (
                "hidden_states",
                [
                    rbln_batch_size,
                    (num_frames - 1) // vae_scale_factor_temporal + 1,
                    model_config.out_channels,
                    sample_size[0],
                    sample_size[1],
                ],
                "float32",
            ),
            (
                "encoder_hidden_states",
                [
                    rbln_batch_size,
                    rbln_max_sequence_length,
                    model_config.text_embed_dim,
                ],
                "float32",
            ),
            ("timestep", [rbln_batch_size], "int64"),
        ]

        rbln_compile_config = RBLNCompileConfig(input_info=input_info)

        rbln_config = RBLNConfig(
            rbln_cls=cls.__name__,
            compile_cfgs=[rbln_compile_config],
            rbln_kwargs=rbln_kwargs,
        )

        rbln_config.model_cfg.update({"batch_size": rbln_batch_size})
        rbln_config.model_cfg.update({"num_frames": num_frames})

        return rbln_config
    
    @property
    def compiled_batch_size(self):
        return self.rbln_config.compile_cfgs[0].input_info[0][1][0]

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: Union[int, float, torch.LongTensor],
        timestep_cond: Optional[torch.Tensor] = None,
        ofs: Optional[Union[int, float, torch.LongTensor]] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ):
        sample_batch_size = hidden_states.size()[0]
        compiled_batch_size = self.compiled_batch_size
        if sample_batch_size != compiled_batch_size and (
            sample_batch_size * 2 == compiled_batch_size or sample_batch_size == compiled_batch_size * 2
        ):
            raise ValueError(
                f"Mismatch between Transformers' runtime batch size ({sample_batch_size}) and compiled batch size ({compiled_batch_size}). "
                "This may be caused by the 'guidance scale' parameter, which doubles the runtime batch size in Stable Diffusion. "
                "Adjust the batch size during compilation or modify the 'guidance scale' to match the compiled batch size.\n\n"
                "For details, see: https://docs.rbln.ai/software/optimum/model_api.html#stable-diffusion"
            )
        sample = super().forward(hidden_states, 
                                 encoder_hidden_states, 
                                 timestep.contiguous())

        if not return_dict:
            return (sample,)
        return Transformer2DModelOutput(sample=sample)
