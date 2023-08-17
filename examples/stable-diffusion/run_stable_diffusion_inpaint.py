import os

import fire
import torch
from diffusers.utils import load_image

from optimum.rbln import RBLNStableDiffusionInpaintPipeline


def main(
    model_id: str = "runwayml/stable-diffusion-inpainting",
    from_diffusers: bool = False,
    prompt: str = "concept art digital painting of an elven castle, inspired by lord of the rings, highly detailed, 8k",
):
    img_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint.png"
    mask_url = (
        "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint_mask.png"
    )
    source = load_image(img_url)
    mask = load_image(mask_url)

    if from_diffusers:
        pipe = RBLNStableDiffusionInpaintPipeline.from_pretrained(
            model_id=model_id,
            export=True,
            rbln_guidance_scale=7.5,
        )
        pipe.save_pretrained(os.path.basename(model_id))
    else:
        pipe = RBLNStableDiffusionInpaintPipeline.from_pretrained(model_id=os.path.basename(model_id), export=False)

    image = pipe(prompt, image=source, mask_image=mask, generator=torch.manual_seed(42)).images[0]
    image.save(f"{prompt}.png")


if __name__ == "__main__":
    fire.Fire(main)
