import os
from typing import Union

import fire
import numpy as np
import torch
from cosmos_guardrail import CosmosSafetyChecker
from diffusers.utils import export_to_video

from optimum.rbln import RBLNCosmosTextToWorldPipeline, RBLNCosmosTextToWorldPipelineConfig


class RBLNMockSafetyChecker(CosmosSafetyChecker):
    def __init__(self):
        torch.nn.Module.__init__(self)

    def check_text_safety(self, prompt: str) -> bool:
        return True

    def check_video_safety(self, frames: np.ndarray) -> np.ndarray:
        return frames

    def to(self, device: Union[str, torch.device] = None, dtype: torch.dtype = None) -> None: ...

    @property
    def device(self) -> torch.device:
        return torch.device("cpu")

    @property
    def dtype(self) -> torch.dtype:
        return torch.float32


def main(
    model_id: str = "nvidia/Cosmos-1.0-Diffusion-7B-Text2World",
    from_diffusers: bool = False,
    prompt: str = None,
    steps: int = 36,
    height: int = 704,
    width: int = 1280,
):
    if prompt is None:
        prompt = "A sleek, humanoid robot stands in a vast warehouse filled with neatly stacked cardboard boxes on industrial shelves. The robot's metallic body gleams under the bright, even lighting, highlighting its futuristic design and intricate joints. A glowing blue light emanates from its chest, adding a touch of advanced technology. The background is dominated by rows of boxes, suggesting a highly organized storage system. The floor is lined with wooden pallets, enhancing the industrial setting. The camera remains static, capturing the robot's poised stance amidst the orderly environment, with a shallow depth of field that keeps the focus on the robot while subtly blurring the background for a cinematic effect."
    print(prompt)

    # from optimum.rbln import RBLNAutoencoderKLCosmos

    # ae = RBLNAutoencoderKLCosmos.from_pretrained(
    #     model_id,
    #     export=True,
    #     subfolder="vae",
    #     rbln_config={
    #         "num_channels_latents": 16,
    #         "vae_scale_factor_temporal": 1,
    #         "vae_scale_factor_spatial": 1,
    #     },
    # )
    # breakpoint()

    # return
    if from_diffusers:
        pipe = RBLNCosmosTextToWorldPipeline.from_pretrained(
            model_id,
            export=True,
            rbln_height=height,
            rbln_width=width,
            rbln_config={
                "transformer": {
                    "device": [4],
                    "tensor_parallel_size": 1,
                },
                "text_encoder": {
                    "device": 1,
                },
                "vae": {
                    "device": 2,
                },
                "safety_checker": {
                    "device": 3,
                    "aegis": {
                        "device": [0, 1, 2, 3],
                    },
                },
            },
            # safety_checker=RBLNMockSafetyChecker(),
        )
        pipe.save_pretrained(os.path.basename(model_id) + "_compiled")
    else:
        pipe = RBLNCosmosTextToWorldPipeline.from_pretrained(
            os.path.basename(model_id),
            export=False,
            rbln_config={
                "transformer": {
                    "device": [4],
                    "tensor_parallel_size": 4,
                },
                "text_encoder": {
                    "device": 1,
                },
                "vae": {
                    "device": 2,
                },
                # TODO how can we support?
                # "safety_checker":{
                #     "device":[0, 1, 2, 3]
                # }
            },
            safety_checker=RBLNMockSafetyChecker(),
        )

    output = pipe(prompt=prompt, num_inference_steps=steps).frames[0]
    export_to_video(output, "output.mp4", fps=30)


if __name__ == "__main__":
    fire.Fire(main)
