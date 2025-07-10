import requests
from PIL import Image
from transformers import CLIPProcessor, CLIPVisionModel, CLIPVisionModelWithProjection

from optimum.rbln import RBLNCLIPVisionModel, RBLNCLIPVisionModelWithProjection


model_p = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True)

output_hidden_states = True
output_attentions = True

rbln_clip = RBLNCLIPVisionModel.from_model(
    model,
    rbln_image_size=(224, 224),
    rbln_output_hidden_states=output_hidden_states,
    rbln_output_attentions=output_attentions,
)
rbln_clip.save_pretrained("clip_model")

rbln_clip_p = RBLNCLIPVisionModelWithProjection.from_model(
    model_p,
    rbln_image_size=(224, 224),
    rbln_output_hidden_states=output_hidden_states,
    rbln_output_attentions=output_attentions,
)
rbln_clip_p.save_pretrained("clip_p_model")

# rbln_clip = RBLNCLIPVisionModel.from_pretrained("clip_model")
# rbln_clip_p = RBLNCLIPVisionModelWithProjection.from_pretrained("clip_p_model")

return_dict = True


import torch


with torch.no_grad():
    outputs_p = model_p(
        inputs["pixel_values"],
        return_dict=return_dict,
        output_hidden_states=output_hidden_states,
        output_attentions=output_attentions,
    )
    outputs_rbln_p = rbln_clip_p(
        inputs["pixel_values"],
        return_dict=return_dict,
        output_hidden_states=output_hidden_states,
        output_attentions=output_attentions,
    )

    outputs = model(
        inputs["pixel_values"],
        return_dict=return_dict,
        output_hidden_states=output_hidden_states,
        output_attentions=output_attentions,
    )
    outputs_rbln = rbln_clip(
        inputs["pixel_values"],
        return_dict=return_dict,
        output_hidden_states=output_hidden_states,
        output_attentions=output_attentions,
    )

import pdb


pdb.set_trace()


from scipy.stats import pearsonr


for i in range(len(outputs_rbln_p)):
    for j in range(len(outputs_p[i])):
        p = pearsonr(outputs_p[i][j].flatten(), outputs_rbln_p[i][j].flatten()).statistic
        print(p)
    print("-" * 20)

print("*" * 50)
for i in range(len(outputs_rbln_p)):
    for j in range(len(outputs_p[i])):
        p = pearsonr(outputs[i][j].flatten(), outputs_rbln[i][j].flatten()).statistic
        print(p)
    print("-" * 20)
