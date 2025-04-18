import os

import cv2
import fire
import numpy as np
import torch
from diffusers import ControlNetModel, UniPCMultistepScheduler
from diffusers.utils import load_image
from PIL import Image

from optimum.rbln import RBLNStableDiffusionControlNetPipeline


def main(
    diffusion_model_id: str = "runwayml/stable-diffusion-v1-5",
    from_diffusers: bool = False,
    controlnet_model_id: str = "lllyasviel/sd-controlnet-canny",
    prompt: str = "the mona lisa",
):
    controlnet = ControlNetModel.from_pretrained(controlnet_model_id)

    image = load_image(
        "https://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/input_image_vermeer.png"
    )

    # input image preprocessing
    np_image = np.array(image)
    np_image = cv2.Canny(np_image, 100, 200)
    np_image = np_image[:, :, None]
    np_image = np.concatenate([np_image, np_image, np_image], axis=2)
    canny_image = Image.fromarray(np_image)

    img_width, img_height = image.size

    if from_diffusers:
        pipe = RBLNStableDiffusionControlNetPipeline.from_pretrained(
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
    else:
        pipe = RBLNStableDiffusionControlNetPipeline.from_pretrained(
            model_id=os.path.basename(diffusion_model_id),
            export=False,
        )

    image = pipe(
        prompt=prompt,
        image=canny_image,
        generator=torch.manual_seed(42),
    ).images[0]

    image.save(f"{prompt}.png")


if __name__ == "__main__":
    fire.Fire(main)
