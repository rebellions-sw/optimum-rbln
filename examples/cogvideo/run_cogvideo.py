import torch
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video
import os
prompt = "A panda, dressed in a small, red jacket and a tiny hat, sits on a wooden stool in a serene bamboo forest. The panda's fluffy paws strum a miniature acoustic guitar, producing soft, melodic tunes. Nearby, a few other pandas gather, watching curiously and some clapping in rhythm. Sunlight filters through the tall bamboo, casting a gentle glow on the scene. The panda's face is expressive, showing concentration and joy as it plays. The background includes a small, flowing stream and vibrant green foliage, enhancing the peaceful and magical atmosphere of this unique musical performance."

from optimum.rbln import RBLNCogVideoXPipeline
model_id = "THUDM/CogVideoX-2b"
pipe = RBLNCogVideoXPipeline.from_pretrained(
    model_id=model_id,
    export=True,
)
pipe.save_pretrained(os.path.basename(model_id))

# pipe = CogVideoXPipeline.from_pretrained(
#     "THUDM/CogVideoX-2b",
#     torch_dtype=torch.float32
# )

# video = pipe(
#     prompt=prompt,
#     num_videos_per_prompt=1,
#     num_inference_steps=1,
#     num_frames=49,
#     guidance_scale=6,
#     generator=torch.manual_seed(42),
# ).frames[0]

# export_to_video(video, "output.mp4", fps=8)

pipe = RBLNCogVideoXPipeline.from_pretrained(
    model_id=model_id,
    export=True,
)
pipe.save_pretrained(os.path.basename(model_id))

# video = pipe(
#     prompt=prompt,
#     num_videos_per_prompt=1,
#     num_inference_steps=1,
#     num_frames=49,
#     guidance_scale=6,
#     generator=torch.manual_seed(42),
# ).frames[0]

# export_to_video(video, "output.mp4", fps=8)