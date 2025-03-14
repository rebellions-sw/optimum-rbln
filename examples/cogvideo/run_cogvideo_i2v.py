import os

import torch
from diffusers.utils import export_to_video, load_image

from optimum.rbln import RBLNCogVideoXImageToVideoPipeline

prompt = "A little girl is riding a bicycle at high speed. Focused, detailed, realistic."
image = load_image(
    "https://huggingface.co/datasets/a-r-r-o-w/tiny-meme-dataset-captioned/resolve/main/images/8.png"
)
image = image.resize((1360, 768)).convert("RGB") # Width / Height

# rbln compile & run
model_id = "THUDM/CogVideoX-5b-I2V"
pipe = RBLNCogVideoXImageToVideoPipeline.from_pretrained(
    model_id=model_id,
    export=True,
)
pipe.save_pretrained(os.path.basename(model_id))
num_frames = 49
video = pipe(
    prompt=prompt,
    num_videos_per_prompt=1,
    num_inference_steps=50,
    num_frames=num_frames,
    guidance_scale=6,
    generator=torch.manual_seed(42),
).frames[0]

export_to_video(video, f"i2v_output_{num_frames}.mp4", fps=8)
