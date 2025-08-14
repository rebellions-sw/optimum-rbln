import requests

import torch
from PIL import Image
from transformers import AutoProcessor, GroundingDinoForObjectDetection
from torch.nn.functional import pad


model_id = "IDEA-Research/grounding-dino-tiny"
device = "cuda" if torch.cuda.is_available() else "cpu"

processor = AutoProcessor.from_pretrained(model_id)
model = GroundingDinoForObjectDetection.from_pretrained(model_id).to(device)


url = "http://images.cocodataset.org/val2017/000000039769.jpg"
url2 = "http://images.cocodataset.org/val2017/000000000139.jpg"
image = Image.open(requests.get(url, stream=True).raw)
image2 = Image.open(requests.get(url2, stream=True).raw)

text = "a cat."
inputs = processor(images=image, text=text, padding="max_length", return_tensors="pt").to(device)
inputs["pixel_values"] = pad(
    inputs["pixel_values"],
    (0, 1333 - inputs["pixel_values"].shape[-1], 0, 1333 - inputs["pixel_values"].shape[-2]),
    value=0,
)
inputs["pixel_mask"] = pad(
    inputs["pixel_mask"], (0, 1333 - inputs["pixel_mask"].shape[-1], 0, 1333 - inputs["pixel_mask"].shape[-2]), value=0
)
breakpoint()
with torch.no_grad():
    outputs = model(**inputs)

results = processor.post_process_grounded_object_detection(
    outputs, inputs.input_ids, box_threshold=0.4, text_threshold=0.3, target_sizes=[image.size[::-1]]
)
