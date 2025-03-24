import os

import fire
import numpy as np
import torch
from diffusers.utils import load_image

from optimum.rbln import RBLNKandinskyV22InpaintCombinedPipeline


def main(
    model_id: str = "kandinsky-community/kandinsky-2-2-decoder-inpaint",
    from_diffusers: bool = False,
    prompt: str = "a hat",
):
    img_url = "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/kandinsky/cat.png"
    init_image = load_image(img_url)

    if from_diffusers:
        pipe = RBLNKandinskyV22InpaintCombinedPipeline.from_pretrained(
            model_id=model_id,
            export=True,
            rbln_img_height=768,
            rbln_img_width=768,
        )
        pipe.save_pretrained(os.path.basename(model_id))
    else:
        pipe = RBLNKandinskyV22InpaintCombinedPipeline.from_pretrained(
            model_id=os.path.basename(model_id), export=False
        )

    generator = torch.manual_seed(42)
    # Mask out the desired area to inpaint
    # In this example, we will draw a hat on the cat's head
    mask = np.zeros((768, 768), dtype=np.float32)
    mask[:250, 250:-250] = 1

    image = pipe(prompt, image=init_image, mask_image=mask, generator=generator).images[0]
    image.save(f"{prompt}.png")


if __name__ == "__main__":
    fire.Fire(main)
