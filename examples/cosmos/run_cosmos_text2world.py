import os
from functools import partial
from unittest.mock import patch

import fire
import torch
from diffusers.pipelines.cosmos.cosmos_guardrail import CosmosSafetyChecker
from diffusers.utils import export_to_video

from optimum.rbln import RBLNCosmosPipeline, RBLNCosmosSafetyChecker


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

    safety_checker_dir = "cosmos_safety_checker"
    with patch("torch.load", partial(torch.load, weights_only=True, map_location=torch.device("cpu"))):
        model = CosmosSafetyChecker()

    if from_diffusers:
        checker = RBLNCosmosSafetyChecker.compile_submodules(
            model=model,
            model_save_dir=safety_checker_dir,
            rbln_height=height,
            rbln_width=width,
            rbln_config={
                "text_guardrail": {
                    "device": [
                        0,
                        1,
                        2,
                        3,
                        4,
                        5,
                        6,
                    ]
                },
                "video_guardrail": {"device": 3},
            },
        )
        pipe = RBLNCosmosPipeline.from_pretrained(
            model_id,
            safety_checker=checker,
            export=True,
            rbln_height=height,
            rbln_width=width,
            rbln_config={
                "transformer": {
                    "device": [0, 1, 3, 4],
                    "tensor_parallel_size": 4,
                },
                "text_encoder": {
                    "device": 4,
                },
                "vae": {
                    "device": 2,
                },
            },
        )
        pipe.save_pretrained(os.path.basename(model_id))
    else:
        checker = RBLNCosmosSafetyChecker.load_submodules(
            model=model,
            model_save_dir=safety_checker_dir,
            rbln_height=height,
            rbln_width=width,
            rbln_config={
                "text_guardrail": {
                    "device": [
                        0,
                        1,
                        2,
                        3,
                    ]
                },
                "video_guardrail": {"device": 3},
            },
        )
        pipe = RBLNCosmosPipeline.from_pretrained(
            model_id,
            safety_checker=checker,
            export=False,
            rbln_height=height,
            rbln_width=width,
            rbln_config={
                "transformer": {
                    "device": [0, 1, 3, 4],
                    "tensor_parallel_size": 4,
                },
                "text_encoder": {
                    "device": 4,
                },
                "vae": {
                    "device": 2,
                },
            },
        )

    output = pipe(prompt=prompt, num_inference_steps=steps).frames[0]
    export_to_video(output, "output.mp4", fps=30)


if __name__ == "__main__":
    fire.Fire(main)
