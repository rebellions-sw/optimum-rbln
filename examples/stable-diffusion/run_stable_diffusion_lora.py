import os

import fire

from optimum.rbln import RBLNStableDiffusionPipeline


def main(
    model_id: str = "Lykon/dreamshaper-7",
    from_diffusers: bool = False,
    prompt: str = "Self-portrait oil painting, a beautiful cyborg with golden hair, 8k",
    steps: int = 4,
):
    if from_diffusers:
        pipe = RBLNStableDiffusionPipeline.from_pretrained(
            model_id=model_id, export=True, lora_ids="latent-consistency/lcm-lora-sdv1-5", rbln_guidance_scale=0.0
        )
        pipe.save_pretrained(os.path.basename(model_id))

    else:
        pipe = RBLNStableDiffusionPipeline.from_pretrained(model_id=os.path.basename(model_id), export=False)

    image = pipe(prompt, num_inference_steps=steps, guidance_scale=0).images[0]

    image.save(f"{prompt}_{steps}.png")


if __name__ == "__main__":
    fire.Fire(main)
