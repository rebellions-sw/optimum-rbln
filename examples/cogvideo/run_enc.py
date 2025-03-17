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
    def __init__(self, model):
        super().__init__()
        self.model = model
        for n, m in self.model.named_modules():
            from diffusers.models.autoencoders.autoencoder_kl_cogvideox import CogVideoXDownBlock3D
            from optimum.rbln.diffusers.models.downsampling import RBLNCogVideoXDownsample3D
            if isinstance(m, CogVideoXDownBlock3D) and m.downsamplers is not None :
                m.downsamplers[0] = RBLNCogVideoXDownsample3D(m.downsamplers[0])

    def forward(self, x):
        h = self.model._encode(x)
        return h
    
import rebel
compiled_model = rebel.compile_from_torch(wrapper(model), vae_enc_input_info)
compiled_model.save("vae_enc.rbln")

print("compile done")

module = rebel.Runtime("vae_enc.rbln", tensor_type="pt")

input = torch.randn(vae_enc_input_info[0][1])
output = module(input)

with torch.no_grad():
    output_pt = wrapper(model)(input)

from scipy import stats

correlation, p_value = stats.pearsonr(output.numpy().flatten(), output_pt.numpy().flatten())
print(correlation)
# output-output_pt