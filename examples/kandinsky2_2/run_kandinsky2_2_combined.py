import os

import fire
import torch

from optimum.rbln import RBLNKandinskyV22CombinedPipeline


def main(
    model_id: str = "kandinsky-community/kandinsky-2-2-decoder",
    from_diffusers: bool = False,
    prompt: str = "red cat, 4k photo",
):
    if from_diffusers:
        pipe = RBLNKandinskyV22CombinedPipeline.from_pretrained(
            model_id=model_id,
            export=True,
            rbln_img_height=768,
            rbln_img_width=768,
        )
        pipe.save_pretrained(os.path.basename(model_id))
    else:
        pipe = RBLNKandinskyV22CombinedPipeline.from_pretrained(model_id=os.path.basename(model_id), export=False)

    generator = torch.manual_seed(42)
    image = pipe(prompt, height=768, width=768, num_inference_steps=50, generator=generator).images[0]
    image.save(f"{prompt}.png")


if __name__ == "__main__":
    fire.Fire(main)
