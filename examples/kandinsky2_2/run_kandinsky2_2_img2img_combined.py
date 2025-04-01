import os

import fire
import torch
from diffusers.utils import load_image

from optimum.rbln import RBLNKandinskyV22Img2ImgCombinedPipeline


def main(
    model_id: str = "kandinsky-community/kandinsky-2-2-decoder",
    from_diffusers: bool = False,
    prompt: str = "A red cartoon frog, 4k",
):
    img_url = "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/kandinsky/frog.png"
    init_image = load_image(img_url)

    if from_diffusers:
        pipe = RBLNKandinskyV22Img2ImgCombinedPipeline.from_pretrained(
            model_id=model_id,
            export=True,
            rbln_img_height=768,
            rbln_img_width=768,
        )
        pipe.save_pretrained(os.path.basename(model_id))
    else:
        pipe = RBLNKandinskyV22Img2ImgCombinedPipeline.from_pretrained(
            model_id=os.path.basename(model_id), export=False
        )

    generator = torch.manual_seed(42)

    image = pipe(
        prompt=prompt,
        image=init_image,
        height=768,
        width=768,
        num_inference_steps=100,
        strength=0.2,
        generator=generator,
    ).images[0]
    image.save(f"{prompt}.png")


if __name__ == "__main__":
    fire.Fire(main)
