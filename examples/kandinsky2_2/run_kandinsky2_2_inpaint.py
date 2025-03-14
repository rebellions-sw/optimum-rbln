import os

import fire
import numpy as np
import torch
from diffusers.utils import load_image

from optimum.rbln import RBLNKandinskyV22InpaintPipeline, RBLNKandinskyV22PriorPipeline


def main(
    prior_model_id: str = "kandinsky-community/kandinsky-2-2-prior",
    inpaint_model_id: str = "kandinsky-community/kandinsky-2-2-decoder-inpaint",
    from_diffusers: bool = False,
    prompt: str = "a hat",
):
    img_url = "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/kandinsky/cat.png"
    init_image = load_image(img_url)

    if from_diffusers:
        prior_pipe = RBLNKandinskyV22PriorPipeline.from_pretrained(
            model_id=prior_model_id,
            export=True,
        )
        prior_pipe.save_pretrained(os.path.basename(prior_model_id))

        pipe = RBLNKandinskyV22InpaintPipeline.from_pretrained(
            model_id=inpaint_model_id,
            export=True,
            rbln_img_width=768,
            rbln_img_height=768,
        )
        pipe.save_pretrained(os.path.basename(inpaint_model_id))
    else:
        prior_pipe = RBLNKandinskyV22PriorPipeline.from_pretrained(
            model_id=os.path.basename(prior_model_id),
            export=False,
        )
        pipe = RBLNKandinskyV22InpaintPipeline.from_pretrained(
            model_id=os.path.basename(inpaint_model_id),
            export=False,
        )

    generator = torch.manual_seed(42)
    image_emb, zero_image_emb = prior_pipe(prompt, generator=generator, return_dict=False)

    # Mask out the desired area to inpaint
    # In this example, we will draw a hat on the cat's head
    mask = np.zeros((768, 768), dtype=np.float32)
    mask[:250, 250:-250] = 1

    out = pipe(
        image=init_image,
        mask_image=mask,
        image_embeds=image_emb,
        negative_image_embeds=zero_image_emb,
        generator=generator,
    )
    image = out.images[0]
    image.save(f"{prompt}.png")


if __name__ == "__main__":
    fire.Fire(main)
