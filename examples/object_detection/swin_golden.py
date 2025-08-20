import os

import fire
import requests
from PIL import Image
from scipy import stats
from transformers import AutoBackbone, AutoImageProcessor

from optimum.rbln import RBLNSwinBackbone


def main(
    model_id: str = "microsoft/swin-tiny-patch4-window7-224",
    from_transformers: bool = False,
):
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    processor = AutoImageProcessor.from_pretrained("shi-labs/nat-mini-in1k-224")
    model = AutoBackbone.from_pretrained(
        "microsoft/swin-tiny-patch4-window7-224", out_features=["stage1", "stage2", "stage3", "stage4"]
    )

    inputs = processor(image, return_tensors="pt")
    outputs = model(**inputs)
    feature_maps = outputs.feature_maps

    if from_transformers:
        rbln_model = RBLNSwinBackbone.from_pretrained(
            model_id=model_id,
            out_features=["stage1", "stage2", "stage3", "stage4"],
            export=True,
        )
        rbln_model.save_pretrained(os.path.basename(model_id))
    else:
        rbln_model = RBLNSwinBackbone.from_pretrained(model_id=os.path.basename(model_id), export=False)

    output = rbln_model(**inputs)

    print(stats.pearsonr(output.feature_maps[0].flatten(), feature_maps[0].detach().numpy().flatten()))
    print(stats.pearsonr(output.feature_maps[1].flatten(), feature_maps[1].detach().numpy().flatten()))
    print(stats.pearsonr(output.feature_maps[2].flatten(), feature_maps[2].detach().numpy().flatten()))
    print(stats.pearsonr(output.feature_maps[3].flatten(), feature_maps[3].detach().numpy().flatten()))


if __name__ == "__main__":
    fire.Fire(main)
