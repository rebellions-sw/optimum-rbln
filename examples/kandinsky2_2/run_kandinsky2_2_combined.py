import os

import fire
import torch

from optimum.rbln import RBLNKandinskyV22CombinedPipeline


def main(
    model_id: str = "kandinsky-community/kandinsky-2-2-decoder",
    from_diffusers: bool = False,
    prompt: str = "A lion in galaxies, spirals, nebulae, stars, smoke, iridescent, intricate detail, octane render, 8k",
):
    if from_diffusers:
        pipe = RBLNKandinskyV22CombinedPipeline.from_pretrained(
            model_id=model_id,
            export=True,
        )
        pipe.save_pretrained(os.path.basename(model_id))
    else:
        pipe = RBLNKandinskyV22CombinedPipeline.from_pretrained(
            model_id=os.path.basename(model_id),
            export=False
        )

    image = pipe(prompt, num_inference_steps=25, generator=torch.manual_seed(42)).images[0]
    image.save(f"{prompt}.png")


if __name__ == "__main__":
    fire.Fire(main)
