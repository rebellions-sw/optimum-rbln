import os

import fire
import torch
from diffusers.utils import load_image

from optimum.rbln import RBLNKandinskyV22Pipeline, RBLNKandinskyV22PriorPipeline


def main(
    prior_model_id: str = "kandinsky-community/kandinsky-2-2-prior",
    inpaint_model_id: str = "kandinsky-community/kandinsky-2-2-decoder",
    from_diffusers: bool = False,
):
    img1 = load_image(
        "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/kandinsky/cat.png"
    )
    img2 = load_image(
        "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/kandinsky/starry_night.jpeg"
    )

    if from_diffusers:
        prior_pipe = RBLNKandinskyV22PriorPipeline.from_pretrained(
            model_id=prior_model_id,
            export=True,
        )
        prior_pipe.save_pretrained(os.path.basename(prior_model_id))

        pipe = RBLNKandinskyV22Pipeline.from_pretrained(
            model_id=inpaint_model_id,
            export=True,
            rbln_img_height=768,
            rbln_img_width=768,
        )
        pipe.save_pretrained(os.path.basename(inpaint_model_id))
    else:
        prior_pipe = RBLNKandinskyV22PriorPipeline.from_pretrained(
            model_id=os.path.basename(prior_model_id),
            export=False,
        )
        pipe = RBLNKandinskyV22Pipeline.from_pretrained(
            model_id=os.path.basename(inpaint_model_id),
            export=False,
        )

    images_texts = ["a cat", img1, img2]
    weights = [0.3, 0.3, 0.4]
    generator = torch.manual_seed(42)
    out = prior_pipe.interpolate(images_texts, weights, generator=generator)
    image_emb = out.image_embeds
    zero_image_emb = out.negative_image_embeds

    out = pipe(
        image_embeds=image_emb,
        negative_image_embeds=zero_image_emb,
        num_inference_steps=50,
        generator=generator,
    )
    image = out.images[0]
    image.save("starry_cat.png")


if __name__ == "__main__":
    fire.Fire(main)
