import os

import fire
import numpy as np
import torch
from diffusers import ControlNetModel, UniPCMultistepScheduler
from diffusers.utils import load_image
from transformers import pipeline

from optimum.rbln import RBLNDPTForDepthEstimation, RBLNStableDiffusionControlNetImg2ImgPipeline


def main(
    diffusion_model_id: str = "runwayml/stable-diffusion-v1-5",
    from_diffusers: bool = False,
    controlnet_model_id: str = "lllyasviel/control_v11f1p_sd15_depth",
    depth_estimator_model_id: str = "Intel/dpt-large",
    prompt: str = "lego batman and robin",
):
    controlnet = ControlNetModel.from_pretrained(controlnet_model_id)

    image = load_image(
        "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/controlnet-img2img.jpg"
    )

    def get_depth_map(image, depth_estimator):
        image = depth_estimator(image)["depth"]
        image = np.array(image)
        image = image[:, :, None]
        image = np.concatenate([image, image, image], axis=2)
        detected_map = torch.from_numpy(image).float() / 255.0
        depth_map = detected_map.permute(2, 0, 1)
        return depth_map

    img_width, img_height = image.size

    if from_diffusers:
        pipe = RBLNStableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            model_id=diffusion_model_id,
            controlnet=controlnet,
            rbln_img_width=img_width,
            rbln_img_height=img_height,
            rbln_config={
                "unet": {
                    "batch_size": 2,
                },
                "controlnet": {
                    "batch_size": 2,
                },
            },
            export=True,
            scheduler=UniPCMultistepScheduler.from_pretrained(diffusion_model_id, subfolder="scheduler"),
        )
        pipe.save_pretrained(os.path.basename(diffusion_model_id))
        de_model = RBLNDPTForDepthEstimation.from_pretrained(
            model_id=depth_estimator_model_id,
            export=True,
        )
        de_model.save_pretrained(os.path.basename(depth_estimator_model_id))
    else:
        pipe = RBLNStableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            model_id=os.path.basename(diffusion_model_id),
            export=False,
        )
        de_model = RBLNDPTForDepthEstimation.from_pretrained(
            model_id=os.path.basename(depth_estimator_model_id),
            export=False,
        )

    depth_estimator = pipeline("depth-estimation", model=de_model, image_processor=depth_estimator_model_id)
    depth_map = get_depth_map(image, depth_estimator).unsqueeze(0)

    image = pipe(
        prompt=prompt,
        image=image,
        control_image=depth_map,
        generator=torch.manual_seed(42),
    ).images[0]

    image.save(f"{prompt}-img2img.png")


if __name__ == "__main__":
    fire.Fire(main)
