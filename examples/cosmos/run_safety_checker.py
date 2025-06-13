from functools import partial
from unittest.mock import patch

import numpy as np
import torch
from cosmos_guardrail import CosmosSafetyChecker
from diffusers.utils.loading_utils import load_video
from diffusers.utils import export_to_video
from diffusers.video_processor import VideoProcessor

from optimum.rbln.diffusers.pipelines.cosmos.cosmos_guardrail import RBLNCosmosSafetyChecker


if __name__ == "__main__":
    vid_processor = VideoProcessor(vae_scale_factor=8)
    height = 704
    width = 1280
    # video = torch.randn(1, 3, 121, 704, 1280, dtype=torch.float32)
    violent = "violent_video.mp4"
    face = "face_video.mp4"
    video = load_video(face)
    video = vid_processor.preprocess_video(video)

    with patch("torch.load", partial(torch.load, weights_only=True, map_location=torch.device("cpu"))):
        model = CosmosSafetyChecker()

    # print("** compile start")
    # checker = RBLNCosmosSafetyChecker.compile_submodules(
    #     model=model,
    #     rbln_height=height,
    #     rbln_width=width,
    #     rbln_config={
    #         "text_guardrail": {
    #             "device": [
    #                 4,
    #                 5,
    #                 6,
    #                 7,
    #             ]
    #         },
    #         "video_guardrail": {"device": 7},
    #         }, model_save_dir="safety_checker"
    # )

    print("** load start")
    checker = RBLNCosmosSafetyChecker.load_submodules(
        model=model,
        rbln_config={
            "text_guardrail": {
                "device": [
                    4,
                    5,
                    6,
                    7,
                ]
            },
            "video_guardrail": {"device": 7},
        },
        model_save_dir="safety_checker",
    )

    video = vid_processor.postprocess_video(video, output_type="np")
    video = (video * 255).astype(np.uint8)
    blurred_video = checker.check_video_safety(video[0])

    if blurred_video is None : 
        print("video is blocked")
        exit(0)
    else :
        video_batch = [blurred_video]
        video = np.stack(video_batch).astype(np.float32) / 255.0 * 2 - 1
        video = torch.from_numpy(video).permute(0, 4, 1, 2, 3)
        video = vid_processor.postprocess_video(video, output_type="np")
        import pdb; pdb.set_trace()
        export_to_video(video[0], "output.mp4", fps=30)
        # prompt = "A sleek, humanoid robot stands in a vast warehouse filled with neatly stacked cardboard boxes on industrial shelves. The robot's metallic body gleams under the bright, even lighting, highlighting its futuristic design and intricate joints. A glowing blue light emanates from its chest, adding a touch of advanced technology. The background is dominated by rows of boxes, suggesting a highly organized storage system. The floor is lined with wooden pallets, enhancing the industrial setting. The camera remains static, capturing the robot's poised stance amidst the orderly environment, with a shallow depth of field that keeps the focus on the robot while subtly blurring the background for a cinematic effect."
        prompt = "naked women"
        output = checker.check_text_safety(prompt)
        if not output:
            print("text is blocked")
