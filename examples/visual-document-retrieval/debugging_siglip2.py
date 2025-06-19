import torch
from PIL import Image
from transformers import ColPaliForRetrieval, ColPaliProcessor

from optimum.rbln import RBLNSiglipVisionModel


golden_model = ColPaliForRetrieval.from_pretrained("vidore/colpali-v1.3-hf")
vision_tower = golden_model.vlm.vision_tower.to(dtype=torch.float32).eval()

# rbln_model = RBLNSiglipVisionModel.from_model(vision_tower, export = True, rbln_output_attentions = True, rbln_output_hidden_states = True)

rbln_model = RBLNSiglipVisionModel.from_pretrained("rbln_vision_tower", export=False)
# rbln_model.save_pretrained("./rbln_vision_tower")


image = Image.new("RGB", (64, 64), color="black")
processor = ColPaliProcessor.from_pretrained("vidore/colpali-v1.3-hf")
pixel_values = processor(images=[image], dtype=torch.float32).pixel_values

with torch.no_grad():
    output = rbln_model(pixel_values)
    vision_tower = vision_tower.to(dtype=torch.float16)
    output_fp16 = vision_tower(pixel_values, output_attentions=True, output_hidden_states=True)
    vision_tower = vision_tower.to(dtype=torch.float32)
    output_fp32 = vision_tower(pixel_values, output_attentions=True, output_hidden_states=True)
    vision_tower = vision_tower.to(dtype=torch.bfloat16)
    output_bf16 = vision_tower(pixel_values, output_attentions=True, output_hidden_states=True)


from scipy.stats import pearsonr


print("rbln vs fp16", pearsonr(output.last_hidden_state.flatten(), output_fp16.last_hidden_state.flatten()).statistic)
print("rbln vs fp32", pearsonr(output.last_hidden_state.flatten(), output_fp32.last_hidden_state.flatten()).statistic)
print(
    "rbln vs bf16",
    pearsonr(
        output.last_hidden_state.flatten(), output_bf16.last_hidden_state.to(dtype=torch.float32).flatten()
    ).statistic,
)
print(
    "fp16 vs fp32",
    pearsonr(output_fp16.last_hidden_state.flatten(), output_fp32.last_hidden_state.flatten()).statistic,
)
print(
    "fp16 vs bf16",
    pearsonr(
        output_fp16.last_hidden_state.flatten(), output_bf16.last_hidden_state.to(dtype=torch.float32).flatten()
    ).statistic,
)
print(
    "fp32 vs bf16",
    pearsonr(
        output_fp32.last_hidden_state.flatten(), output_bf16.last_hidden_state.to(dtype=torch.float32).flatten()
    ).statistic,
)
# breakpoint()

for i, (rbln_attn, fp32_attn) in enumerate(zip(output.attentions, output_fp32.attentions)):
    print(
        f"layer {i}'s hidden state pearsonr: {pearsonr(output.hidden_states[i].numpy().flatten(), output_fp32.hidden_states[i].numpy().flatten()).statistic}"
    )
    print(
        f"layer {i}'s attention pearsonr: {pearsonr(rbln_attn.numpy().flatten(), fp32_attn.numpy().flatten()).statistic}"
    )

print(pearsonr(output.hidden_states[-1].numpy().flatten(), output_fp32.hidden_states[-1].numpy().flatten()).statistic)

breakpoint()
