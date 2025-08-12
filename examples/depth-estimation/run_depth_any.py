import os

import fire
import requests
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import numpy as np
from scipy import stats
from optimum.rbln import RBLNDepthAnythingForDepthEstimation


def main(
    model_id: str = "depth-anything/Depth-Anything-V2-Small-hf",
    from_transformers: bool = False,
):
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    image_processor = AutoImageProcessor.from_pretrained(model_id)

    if from_transformers:
        model = RBLNDepthAnythingForDepthEstimation.from_pretrained(
            model_id=model_id,
            rbln_image_size=(518, 686),
            export=True,
        )
        model.save_pretrained(os.path.basename(model_id))
    else:
        model = RBLNDepthAnythingForDepthEstimation.from_pretrained(model_id=os.path.basename(model_id), export=False)

    inputs = image_processor(images=image, return_tensors="pt")

    outputs = model(**inputs)
    predicted_depth = outputs.predicted_depth
    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=image.size[::-1],
        mode="bicubic",
        align_corners=False,
    )

    base_model = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
    with torch.no_grad():
        golden_outputs = model(**inputs)
        golden_predicted_depth = outputs.predicted_depth
        
    golden_prediction = torch.nn.functional.interpolate(
        golden_predicted_depth.unsqueeze(1),
        size=image.size[::-1],
        mode="bicubic",
        align_corners=False,
    )
    
    print(stats.pearsonr(golden_prediction.detach().numpy().flatten(), prediction.numpy().flatten()))

if __name__ == "__main__":
    fire.Fire(main)
