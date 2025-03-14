import os

import fire
import torch

from optimum.rbln import RBLNKandinskyV22Pipeline, RBLNKandinskyV22PriorPipeline


def main(
    prior_model_id: str = "kandinsky-community/kandinsky-2-2-prior",
    inpaint_model_id: str = "kandinsky-community/kandinsky-2-2-decoder",
    from_diffusers: bool = False,
    prompt: str = "red cat, 4k photo",
):
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

    generator = torch.manual_seed(42)
    out = prior_pipe(prompt, generator=generator)
    image_emb = out.image_embeds
    zero_image_emb = out.negative_image_embeds

    out = pipe(
        image_embeds=image_emb,
        negative_image_embeds=zero_image_emb,
        height=768,
        width=768,
        num_inference_steps=50,
        generator=generator,
    )
    image = out.images[0]
    image.save(f"{prompt}.png")


if __name__ == "__main__":
    fire.Fire(main)
