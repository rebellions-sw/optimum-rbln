import fire
import rebel
import torch
from diffusers import FluxTransformer2DModel


class TransformerBlockWrapper(torch.nn.Module):
    def __init__(self, transformer_block):
        super().__init__()
        self.transformer_block = transformer_block

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb_0: torch.Tensor,
        image_rotary_emb_1: torch.Tensor,
    ):
        image_rotary_emb = (image_rotary_emb_0, image_rotary_emb_1)
        return self.transformer_block(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            temb=temb,
            image_rotary_emb=image_rotary_emb,
        )


class SingleTransformerBlockWrapper(torch.nn.Module):
    def __init__(self, single_transformer_block):
        super().__init__()
        self.single_transformer_block = single_transformer_block

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb_0: torch.Tensor,
        image_rotary_emb_1: torch.Tensor,
    ):
        image_rotary_emb = (image_rotary_emb_0, image_rotary_emb_1)
        return self.single_transformer_block(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            temb=temb,
            image_rotary_emb=image_rotary_emb,
        )


def main(
    model_id: str = "black-forest-labs/FLUX.1-dev",
    subfolder: str = "transformer",
    num_layers: int = 1,
    num_single_layers: int = 1,
    single: bool = False,
):
    model = FluxTransformer2DModel.from_pretrained(
        model_id,
        subfolder=subfolder,
        num_layers=num_layers,
        num_single_layers=num_single_layers,
    )

    if not single:
        transformer_block = model.transformer_blocks[0]
        wrapped_transformer_block = TransformerBlockWrapper(transformer_block)
        compiled_model = rebel.compile_from_torch(
            wrapped_transformer_block,
            input_info=[
                ("hidden_states", [1, 4096, 3072], "float32"),
                ("encoder_hidden_states", [1, 512, 3072], "float32"),
                ("temb", [1, 3072], "float32"),
                ("image_rotary_emb_0", [4608, 128], "float32"),
                ("image_rotary_emb_1", [4608, 128], "float32"),
            ],
        )
    else:
        single_transformer_block = model.single_transformer_blocks[0]
        wrapped_single_transformer_block = TransformerBlockWrapper(single_transformer_block)
        compiled_model = rebel.compile_from_torch(
            wrapped_single_transformer_block,
            input_info=[
                ("hidden_states", [1, 4096, 3072], "float32"),
                ("encoder_hidden_states", [1, 512, 3072], "float32"),
                ("temb", [1, 3072], "float32"),
                ("image_rotary_emb_0", [4608, 128], "float32"),
                ("image_rotary_emb_1", [4608, 128], "float32"),
            ],
        )

    # need to add verification step here (golden vs compiled model)


if __name__ == "__main__":
    fire.Fire(main)
