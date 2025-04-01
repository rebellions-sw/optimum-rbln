import os

import torch
from diffusers.utils import export_to_video

from optimum.rbln import RBLNCogVideoXPipeline


prompt = "A panda, dressed in a small, red jacket and a tiny hat, sits on a wooden stool in a serene bamboo forest. The panda's fluffy paws strum a miniature acoustic guitar, producing soft, melodic tunes. Nearby, a few other pandas gather, watching curiously and some clapping in rhythm. Sunlight filters through the tall bamboo, casting a gentle glow on the scene. The panda's face is expressive, showing concentration and joy as it plays. The background includes a small, flowing stream and vibrant green foliage, enhancing the peaceful and magical atmosphere of this unique musical performance."


# golden result
# golden_pipe = CogVideoXPipeline.from_pretrained(
#     "THUDM/CogVideoX-2b",
#     torch_dtype=torch.float32
# )

# golden_video = golden_pipe(
#     prompt=prompt,
#     num_videos_per_prompt=1,
#     num_inference_steps=50,
#     num_frames=49,
#     guidance_scale=6,
#     generator=torch.manual_seed(42),
# ).frames[0]

# export_to_video(golden_video, "golden_output.mp4", fps=8)


# rbln compile & run
# model_id = "THUDM/CogVideoX-2b"
# model_id = "/home/seinpark/optimum-rbln/examples/cogvideo/CogVideoX-2b_0331"
model_id = "/home/seinpark/nas_data/cogvideox/CogVideoX-2b_0331"
pipe = RBLNCogVideoXPipeline.from_pretrained(
    model_id=os.path.basename(model_id),
    export=False,
    # model_id=model_id,
    # export=True,
)
# pipe.save_pretrained(os.path.basename(model_id))

video = pipe(
    prompt=prompt,
    num_videos_per_prompt=1,
    num_inference_steps=50,
    num_frames=49,
    guidance_scale=6,
    generator=torch.manual_seed(42),
).frames[0]

export_to_video(video, "output.mp4", fps=8)
