# import requests

# import torch
# from PIL import Image
# from transformers import AutoProcessor, GroundingDinoForObjectDetection


# model_id = "IDEA-Research/grounding-dino-tiny"
# device = "cuda" if torch.cuda.is_available() else "cpu"

# processor = AutoProcessor.from_pretrained(model_id)
# model = GroundingDinoForObjectDetection.from_pretrained(model_id).to(device)


# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# url2 = "http://images.cocodataset.org/val2017/000000000139.jpg"
# image = Image.open(requests.get(url, stream=True).raw)
# image2 = Image.open(requests.get(url2, stream=True).raw)

# text = ["a cat.", "a living room."]
# images = [image, image2]

# inputs = processor(images=images, text=text,  padding='max_length',return_tensors="pt").to(device)
# with torch.no_grad():
#     outputs = model(**inputs)

# results = processor.post_process_grounded_object_detection(
#     outputs,
#     inputs.input_ids,
#     box_threshold=0.4,
#     text_threshold=0.3,
#     target_sizes=[image.size[::-1]]
# )


from transformers import AutoProcessor, AutoModel, AutoModelForZeroShotObjectDetection
from PIL import Image
import requests

from optimum.rbln import RBLNGroundingDinoForObjectDetection

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
url2 = "http://images.cocodataset.org/val2017/000000000139.jpg"
image = Image.open(requests.get(url, stream=True).raw)
image2 = Image.open(requests.get(url2, stream=True).raw)

text = ["a cat.", "a living room."]
images = [image, image2]

processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
# model = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-tiny")

# inputs = processor(images=image, text=text, return_tensors="pt")
# max_text_len
inputs = processor(images=images, text=text, padding="max_length", max_length=256, return_tensors="pt")


# outputs = model(**inputs)
# last_hidden_states = outputs.last_hidden_state

rbln_model = RBLNGroundingDinoForObjectDetection.from_pretrained(
    model_id="IDEA-Research/grounding-dino-tiny",
    export=True,
)

# rbln_model(inputs['input_ids'], inputs['token_type_ids'])
