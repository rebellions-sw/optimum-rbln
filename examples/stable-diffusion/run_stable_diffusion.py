import os

import fire
import torch

from optimum.rbln import RBLNStableDiffusionPipeline


def main(
    model_id: str = "runwayml/stable-diffusion-v1-5",
    from_diffusers: bool = False,
    prompt: str = "an illustration of a cute white cat riding a black horse on mars",
    steps: int = 50,
):
    if from_diffusers:
        pipe = RBLNStableDiffusionPipeline.from_pretrained(
            model_id=model_id,
            export=True,
            rbln_config={
                "unet": {
                    "batch_size": 2,
                }
            },
        )
        pipe.save_pretrained(os.path.basename(model_id))
    else:
        pipe = RBLNStableDiffusionPipeline.from_pretrained(model_id=os.path.basename(model_id), export=False)

    image = pipe(
        prompt,
        num_inference_steps=steps,
        generator=torch.manual_seed(42),
    ).images[0]
    image.save(f"{prompt}.png")


if __name__ == "__main__":
    fire.Fire(main)
