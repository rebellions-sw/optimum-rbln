import os

import fire
import torch

from optimum.rbln import RBLNStableDiffusionXLPipeline


def main(
    model_id: str = "stabilityai/stable-diffusion-xl-base-1.0",
    rbln_img_path: str = "rbln_img.png",
    prompt: str = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
):
    rbln_pipe = RBLNStableDiffusionXLPipeline.from_pretrained(
        model_id=model_id,
        rbln_config={"unet": {"batch_size": 2}},
        export=True,
    )
    rbln_pipe.save_pretrained(os.path.basename(model_id))

    rbln_image = rbln_pipe(
        prompt,
        generator=torch.manual_seed(42),
    ).images[0]

    if rbln_image:
        rbln_image.save(rbln_img_path)
    else:
        raise ValueError("RBLN image is None")


if __name__ == "__main__":
    fire.Fire(main)
