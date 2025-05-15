import os

import fire
import torch
from optimum.rbln import RBLNFluxPipeline
from diffusers import FluxPipeline


def main(
    model_id: str = "black-forest-labs/FLUX.1-dev",
    from_diffusers: bool = False,
    prompt: str = "A cat holding a sign that says hello world",
):
    # golden inference
    # original_model = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.float32)
    # image = original_model(prompt, num_inference_steps=2, max_sequence_length=512, guidance_scale=3.5, generator=torch.manual_seed(0)).images[0]
    # image.save("original.png")
    # exit()
    
    if from_diffusers:
        pipe = RBLNFluxPipeline.from_pretrained(
            model_id=model_id,
            export=True,
            rbln_guidance_scale=3.5,
            rbln_batch_size=1,
            rbln_config={
                "transformer" : {
                    "tensor_parallel_size": 4
                }
            }
        )
        pipe.save_pretrained(os.path.basename(model_id))
    else:
        pipe = RBLNFluxPipeline.from_pretrained(
            model_id=os.path.basename(model_id),
            export=False
        )

    image = pipe(prompt, num_inference_steps=2, max_sequence_length=512, guidance_scale=3.5, generator=torch.manual_seed(0)).images[0]
    image.save(f"{prompt}.png")


if __name__ == "__main__":
    fire.Fire(main)