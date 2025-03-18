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

from diffusers import CogVideoXImageToVideoPipeline
model = CogVideoXImageToVideoPipeline.from_pretrained(
    pretrained_model_name_or_path=model_id,
).vae

enc_shape = (480, 720)
vae_enc_input_info = [
    (
        "x",
        [
            1,
            3,
            1,
            enc_shape[0],
            enc_shape[1],
        ],
        "float32",
    )
]

class wrapper(torch.nn.Module):
    def __init__(self, model, hide_op=False):
        super().__init__()
        self.model = self._hide_dead_op(model) if hide_op else model

    def _hide_dead_op(self, model):
        for m in model.modules():
            from diffusers.models.autoencoders.autoencoder_kl_cogvideox import CogVideoXDownBlock3D
            from optimum.rbln.diffusers.models.downsampling import RBLNCogVideoXDownsample3D
            if isinstance(m, CogVideoXDownBlock3D) and m.downsamplers is not None :
                m.downsamplers[0] = RBLNCogVideoXDownsample3D(m.downsamplers[0])
        return model
    
    def forward(self, x):
        h = self.model._encode(x)
        return h
    
import rebel
compiled_model = rebel.compile_from_torch(wrapper(model, hide_op=True), vae_enc_input_info)
compiled_model.save("vae_enc.rbln")

print("compile done")

module = rebel.Runtime("vae_enc.rbln", tensor_type="pt")

input = torch.randn(vae_enc_input_info[0][1])
output = module(input)

model_pt = wrapper(model).eval()
with torch.no_grad():
    output_pt = model_pt(input)

from scipy import stats

correlation, p_value = stats.pearsonr(output.numpy().flatten(), output_pt.numpy().flatten())
print(correlation)
output-output_pt