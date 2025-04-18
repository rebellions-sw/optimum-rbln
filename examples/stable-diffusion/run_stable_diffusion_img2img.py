import os
from io import BytesIO

import fire
import requests
import torch
from PIL import Image

from optimum.rbln import RBLNStableDiffusionImg2ImgPipeline


def main(
    model_id: str = "runwayml/stable-diffusion-v1-5",
    from_diffusers: bool = False,
    prompt: str = "A fantasy landscape, trending on artstation",
    img_width: int = 768,
    img_height: int = 512,
    guidance_scale: float = 7.5,
    strength: float = 0.75,
):
    url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"
    response = requests.get(url)
    init_image = Image.open(BytesIO(response.content)).convert("RGB")
    init_image = init_image.resize((img_width, img_height))

    if from_diffusers:
        pipe = RBLNStableDiffusionImg2ImgPipeline.from_pretrained(
            model_id=model_id,
            export=True,
            rbln_img_width=img_width,
            rbln_img_height=img_height,
            rbln_config={
                "unet": {
                    "batch_size": 2,
                },
            },
        )
        pipe.save_pretrained(os.path.basename(model_id))
    else:
        pipe = RBLNStableDiffusionImg2ImgPipeline.from_pretrained(
            model_id=os.path.basename(model_id),
            export=False,
        )

    image = pipe(
        prompt=prompt,
        image=init_image,
        strength=strength,
        guidance_scale=guidance_scale,
        generator=torch.manual_seed(42),
    ).images[0]

    image.save(f"{prompt}.png")


if __name__ == "__main__":
    fire.Fire(main)
