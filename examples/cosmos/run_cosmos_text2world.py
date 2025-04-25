import os
from functools import partial
from unittest.mock import patch

import fire
import torch
from diffusers.pipelines.cosmos.cosmos_guardrail import CosmosSafetyChecker
from diffusers.utils import export_to_video

from optimum.rbln.diffusers.pipelines.cosmos import RBLNCosmosPipeline, RBLNCosmosSafetyChecker


def main(
    model_id: str = "nvidia/Cosmos-1.0-Diffusion-7B-Text2World",
    from_diffusers: bool = False,
    prompt: str = "an illustration of a cute white cat riding a black horse on mars",
    steps: int = 36,
):
    model_id = "/mnt/shared_data/users/dkhong/nas_data/cosmos_examples/CosmosPredict1"  # FIXME: For test
    if from_diffusers:
        safety_checker_dir = "cosmos_safety_checker"
        with patch("torch.load", partial(torch.load, weights_only=True, map_location=torch.device("cpu"))):
            model = CosmosSafetyChecker()
        checker = RBLNCosmosSafetyChecker.compile_submodules(
            model=model,
            model_save_dir=safety_checker_dir,
            rbln_config={
                "text_guardrail": {
                    "device": [
                        0,
                        1,
                        2,
                        3,
                    ]
                },
                "video_guardrail": {"device": 0},
            },
        )
        pipe = RBLNCosmosPipeline.from_pretrained(model_id, safety_checker=checker, export=True)
        pipe.save_pretrained(os.path.basename(model_id))
    else:
        safety_checker_dir = "cosmos_safety_checker"
        with patch("torch.load", partial(torch.load, weights_only=True, map_location=torch.device("cpu"))):
            model = CosmosSafetyChecker()
        checker = RBLNCosmosSafetyChecker.load_submodules(
            model=model,
            model_save_dir=safety_checker_dir,
            rbln_config={
                "text_guardrail": {
                    "device": [
                        0,
                        1,
                        2,
                        3,
                    ]
                },
                "video_guardrail": {"device": 0},
            },
        )
        pipe = RBLNCosmosPipeline.from_pretrained(model_id, safety_checker=checker, export=False)

    prompt = "A sleek, humanoid robot stands in a vast warehouse filled with neatly stacked cardboard boxes on industrial shelves. The robot's metallic body gleams under the bright, even lighting, highlighting its futuristic design and intricate joints. A glowing blue light emanates from its chest, adding a touch of advanced technology. The background is dominated by rows of boxes, suggesting a highly organized storage system. The floor is lined with wooden pallets, enhancing the industrial setting. The camera remains static, capturing the robot's poised stance amidst the orderly environment, with a shallow depth of field that keeps the focus on the robot while subtly blurring the background for a cinematic effect."

    output = pipe(prompt=prompt, num_inference_steps=steps).frames[0]
    export_to_video(output, "output.mp4", fps=30)


if __name__ == "__main__":
    fire.Fire(main)
