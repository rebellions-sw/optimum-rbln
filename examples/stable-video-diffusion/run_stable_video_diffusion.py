import os
from typing import Optional

import fire
import torch
from diffusers.utils import export_to_video, load_image

from optimum.rbln import RBLNStableVideoDiffusionPipeline


def main(
    model_id: str = "stabilityai/stable-video-diffusion-img2vid-xt-1-1",
    from_diffusers: bool = False,
    img_width: int = 1024,
    img_height: int = 576,
    num_frames: Optional[int] = None,
    decode_chunk_size: Optional[int] = None,
):
    url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/svd/rocket.png"
    image = load_image(url)
    image = image.resize((img_width, img_height))

    if from_diffusers:
        pipe = RBLNStableVideoDiffusionPipeline.from_pretrained(
            model_id,
            export=True,
            rbln_img_width=img_width,
            rbln_img_height=img_height,
            rbln_num_frames=num_frames,
            rbln_decode_chunk_size=decode_chunk_size,
            rbln_config={
                "unet": {"device": 0},
                "image_encoder": {"device": 0},
                "vae": {"device": 1},
            },
        )
        pipe.save_pretrained(os.path.basename(model_id))
    else:
        pipe = RBLNStableVideoDiffusionPipeline.from_pretrained(
            model_id=os.path.basename(model_id),
            export=False,
            rbln_config={
                "unet": {"device": 0},
                "image_encoder": {"device": 0},
                "vae": {"device": 1},
            },
        )

    generator = torch.manual_seed(42)
    frames = pipe(
        image=image,
        height=img_height,
        width=img_width,
        num_frames=num_frames,
        generator=generator,
    ).frames[0]

    if num_frames is None:
        num_frames = pipe.unet.config.num_frames

    export_to_video(frames, f"generated_{os.path.basename(model_id)}.mp4", fps=num_frames)


if __name__ == "__main__":
    fire.Fire(main)
