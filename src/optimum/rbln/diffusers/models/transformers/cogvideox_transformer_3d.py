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
    def __post_init__(self, **kwargs):
        super().__post_init__(**kwargs)

    @classmethod
    def wrap_model_if_needed(cls, model: torch.nn.Module, rbln_config: RBLNConfig) -> torch.nn.Module:
        return CogVideoXTransformer3DModelWrapper(model).eval()

    @classmethod
    def update_rbln_config_using_pipe(cls, pipe: RBLNDiffusionMixin, rbln_config: Dict[str, Any]) -> Dict[str, Any]:
        # sample_size = rbln_config.get("sample_size", pipe.default_sample_size)
        # img_width = rbln_config.get("img_width")
        # img_height = rbln_config.get("img_height")

        # if (img_height is None) ^ (img_width is None):
        #     raise RuntimeError

        # elif img_height and img_width:
        #     sample_size = img_height // pipe.vae_scale_factor, img_width // pipe.vae_scale_factor

        # max_sequence_length = pipe.tokenizer_2.model_max_length
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

        return {
            "batch_size": batch_size,
            # "max_sequence_length": max_sequence_length,
            # "sample_size": sample_size,
            # "vae_scale_factor": pipe.vae_scale_factor,
        }

    @classmethod
    def _get_rbln_config(
        cls,
        preprocessors: Union["AutoFeatureExtractor", "AutoProcessor", "AutoTokenizer"],
        model_config: "PretrainedConfig",
        rbln_kwargs: Dict[str, Any] = {},
    ) -> RBLNConfig:
        rbln_batch_size = rbln_kwargs.get("batch_size", None)

        # sample_size = rbln_kwargs.get("sample_size", None)
        # vae_scale_factor = rbln_kwargs.get("vae_scale_factor", None)

        # if isinstance(sample_size, int):
        #     sample_size = (sample_size, sample_size)

        # rbln_max_seqeunce_length = rbln_kwargs.get("max_sequence_length")
        # if rbln_max_seqeunce_length is None:
        #     raise ValueError("rbln_max_seqeunce_length should be specified.")

        # # prepare_latents function
        # height = 2 * (int(sample_size[0]) // vae_scale_factor)
        # width = 2 * (int(sample_size[1]) // vae_scale_factor)
        # latent_shape = (height // 2) * (width // 2)
        # num_channels_latents = model_config.in_channels // 4


        # hidden_states.shape
        # batch_size, , out_channels, sample_height, sample_width
        # torch.Size([2, 13, 16, 60, 90])
        # encoder_hidden_states.shape
        # batch_size, max_text_seq_length, text_embed_dim
        # torch.Size([2, 226, 4096])
        # timestep.shape
        # torch.Size([2]), tensor([999, 999])
        
        input_info = [
            (
                "hidden_states",
                [
                    rbln_batch_size,
                    #FIXME: for temporal, need generalize
                    13,
                    model_config.out_channels,
                    model_config.sample_height,
                    model_config.sample_width
                ],
                "float32",
            ),
            (
                "encoder_hidden_states",
                [
                    rbln_batch_size,
                    model_config.max_text_seq_length,
                    model_config.text_embed_dim,
                ],
                "float32",
            ),
            ("timestep", [rbln_batch_size], "float32"),

        ]

        rbln_compile_config = RBLNCompileConfig(input_info=input_info)

        rbln_config = RBLNConfig(
            rbln_cls=cls.__name__,
            compile_cfgs=[rbln_compile_config],
            rbln_kwargs=rbln_kwargs,
        )

        rbln_config.model_cfg.update({"batch_size": rbln_batch_size})

        return rbln_config

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
        output = super().forward(hidden_states, encoder_hidden_states, timestep)
        return Transformer2DModelOutput(sample=output)