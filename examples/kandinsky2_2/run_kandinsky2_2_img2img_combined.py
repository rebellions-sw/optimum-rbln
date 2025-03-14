import os
from io import BytesIO

import fire
import requests
import torch
from PIL import Image

from optimum.rbln import RBLNKandinskyV22Img2ImgCombinedPipeline


def main(
    model_id: str = "kandinsky-community/kandinsky-2-2-decoder",
    from_diffusers: bool = False,
    prompt: str = "A fantasy landscape, Cinematic lighting",
    negative_prompt: str = "low quality, bad quality",
    img_width: int = 1024,
    img_height: int = 512,
):
    if from_diffusers:
        pipe = RBLNKandinskyV22Img2ImgCombinedPipeline.from_pretrained(
            model_id=model_id,
            export=True,
            rbln_img_height=img_height,
            rbln_img_width=img_width,
            rbln_config={
                "prior_text_encoder": {
                    "batch_size": 2,
                },
                "prior_prior": {
                    "batch_size": 4,
                },
            },
        )
        pipe.save_pretrained(os.path.basename(model_id))
    else:
        pipe = RBLNKandinskyV22Img2ImgCombinedPipeline.from_pretrained(
            model_id=os.path.basename(model_id),
            export=False
        )

    url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"

    response = requests.get(url)
    init_image = Image.open(BytesIO(response.content)).convert("RGB")
    init_image.thumbnail((img_width, img_height))

    image = pipe(prompt=prompt, image=init_image, negative_prompt=negative_prompt, height=img_height, width=img_width, num_inference_steps=25, generator=torch.manual_seed(42)).images[0]
    image.save(f"{prompt}.png")


if __name__ == "__main__":
    fire.Fire(main)
