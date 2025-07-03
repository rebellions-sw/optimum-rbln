import os

import fire
from diffusers.utils import export_to_video

from optimum.rbln import RBLNCosmosTextToWorldPipeline


def main(
    model_id: str = "nvidia/Cosmos-1.0-Diffusion-7B-Text2World",
    from_diffusers: bool = False,
    prompt: str = None,
    steps: int = 36,
    height: int = 704,
    width: int = 1280,
):
    if prompt is None:
        prompt = "A sleek, humanoid robot stands in a vast warehouse filled with neatly stacked cardboard boxes on industrial shelves. The robot's metallic body gleams under the bright, even lighting, highlighting its futuristic design and intricate joints. A glowing blue light emanates from its chest, adding a touch of advanced technology. The background is dominated by rows of boxes, suggesting a highly organized storage system. The floor is lined with wooden pallets, enhancing the industrial setting. The camera remains static, capturing the robot's poised stance amidst the orderly environment, with a shallow depth of field that keeps the focus on the robot while subtly blurring the background for a cinematic effect."
    print(prompt)

    if from_diffusers:
        pipe = RBLNCosmosTextToWorldPipeline.from_pretrained(
            model_id,
            export=True,
            rbln_height=height,
            rbln_width=width,
            rbln_config={
                "transformer": {
                    "device": [4, 5, 6, 7],
                    "tensor_parallel_size": 4,
                },
                "text_encoder": {
                    "device": 1,
                },
                "vae": {
                    "device": 2,
                },
                "safety_checker": {
                    "aegis": {"device": [0, 1, 2, 3]},
                },
            },
        )
        pipe.save_pretrained(os.path.basename(model_id))
    else:
        pipe = RBLNCosmosTextToWorldPipeline.from_pretrained(
            os.path.basename(model_id),
            export=False,
            rbln_config={
                "transformer": {
                    "device": [4, 5, 6, 7],
                    "tensor_parallel_size": 4,
                },
                "text_encoder": {
                    "device": 1,
                },
                "vae": {
                    "device": 2,
                },
                "safety_checker": {
                    "aegis": {"device": [0, 1, 2, 3]},
                },
            },
        )

    output = pipe(prompt=prompt, num_inference_steps=steps).frames[0]
    export_to_video(output, "output.mp4", fps=30)


if __name__ == "__main__":
    fire.Fire(main)
