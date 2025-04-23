import os

import fire
import torch
from diffusers.utils import load_image

from optimum.rbln import RBLNKandinskyV22Img2ImgPipeline, RBLNKandinskyV22PriorPipeline


def main(
    prior_model_id: str = "kandinsky-community/kandinsky-2-2-prior",
    inpaint_model_id: str = "kandinsky-community/kandinsky-2-2-decoder",
    from_diffusers: bool = False,
    prompt: str = "A red cartoon frog, 4k",
):
    img_url = "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/kandinsky/frog.png"
    init_image = load_image(img_url)

    if from_diffusers:
        prior_pipe = RBLNKandinskyV22PriorPipeline.from_pretrained(
            model_id=prior_model_id,
            export=True,
            rbln_config={"prior": {"batch_size": 2}},
        )
        prior_pipe.save_pretrained(os.path.basename(prior_model_id))

        pipe = RBLNKandinskyV22Img2ImgPipeline.from_pretrained(
            model_id=inpaint_model_id,
            export=True,
            rbln_img_height=768,
            rbln_img_width=768,
            rbln_config={"unet": {"batch_size": 2}},
        )
        pipe.save_pretrained(os.path.basename(inpaint_model_id))
    else:
        prior_pipe = RBLNKandinskyV22PriorPipeline.from_pretrained(
            model_id=os.path.basename(prior_model_id),
            export=False,
        )
        pipe = RBLNKandinskyV22Img2ImgPipeline.from_pretrained(
            model_id=os.path.basename(inpaint_model_id),
            export=False,
        )

    generator = torch.manual_seed(42)
    image_emb, zero_image_emb = prior_pipe(prompt, generator=generator, return_dict=False)

    out = pipe(
        image=init_image,
        image_embeds=image_emb,
        negative_image_embeds=zero_image_emb,
        height=768,
        width=768,
        num_inference_steps=100,
        strength=0.2,
        generator=generator,
    )
    image = out.images[0]
    image.save(f"{prompt}.png")


if __name__ == "__main__":
    fire.Fire(main)
