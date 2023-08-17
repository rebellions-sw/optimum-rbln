import os
import urllib

import fire
import numpy as np
import torch
from PIL import Image
from transformers import DPTImageProcessor

from optimum.rbln import RBLNDPTForDepthEstimation


def main(
    model_id: str = "Intel/dpt-large",
    from_transformers: bool = False,
):
    img_url = "https://rbln-public.s3.ap-northeast-2.amazonaws.com/images/tabby.jpg"
    img_path = "./tabby.jpg"
    if not os.path.exists(img_path):
        with urllib.request.urlopen(img_url) as response, open(img_path, "wb") as f:
            f.write(response.read())
    image = Image.open(img_path)

    processor = DPTImageProcessor.from_pretrained(model_id)

    if from_transformers:
        model = RBLNDPTForDepthEstimation.from_pretrained(
            model_id=model_id,
            export=True,
        )
        model.save_pretrained(os.path.basename(model_id))
    else:
        model = RBLNDPTForDepthEstimation.from_pretrained(model_id=os.path.basename(model_id), export=False)

    inputs = processor(images=image, return_tensors="pt")

    if model.config.is_hybrid:
        predicted_depth = model(**inputs).predicted_depth[0]
    else:
        predicted_depth = model(**inputs).predicted_depth

    # interpolate to original size
    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=image.size[::-1],
        mode="bicubic",
        align_corners=False,
    )

    output = prediction.squeeze().numpy()
    formatted = (output * 255 / np.max(output)).astype("uint8")
    depth = Image.fromarray(formatted)
    depth.save(f"depth_{model_id[6:]}.png")


if __name__ == "__main__":
    fire.Fire(main)
