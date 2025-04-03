import os
import torch
import urllib

import fire
from PIL import Image
from transformers import DetrImageProcessor

from optimum.rbln import RBLNDetrForObjectDetection


def main(
    model_id: str = "facebook/detr-resnet-50",
    from_transformers: bool = False,
    height_size: int = 600,
    width_size: int = 480,
    batch_size: int = 1,
):
    img_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    img_path = "./000000039769.jpg"
    if not os.path.exists(img_path):
        with urllib.request.urlopen(img_url) as response, open(img_path, "wb") as f:
            f.write(response.read())

    image = Image.open(img_path)

    if from_transformers:
        model = RBLNDetrForObjectDetection.from_pretrained(
            model_id,
            export=True,
            rbln_image_size={
                "height": height_size,
                "width": width_size,
            },
            rbln_batch_size=batch_size,
            revision="no_timm",
        )
        model.save_pretrained(os.path.basename(model_id))
    else:
        model = RBLNDetrForObjectDetection.from_pretrained(model_id=os.path.basename(model_id), export=False)

    image_processor = DetrImageProcessor.from_pretrained(model_id)
    inputs = image_processor(
        [image] * batch_size, size={"width": width_size, "height": height_size}, return_tensors="pt"
    )

    outputs = model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]] * batch_size)
    results = image_processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)
    for i in range(batch_size):
        print(f"====== batch {i} ======")
        for score, label, box in zip(results[i]["scores"], results[i]["labels"], results[i]["boxes"]):
            box = [round(i, 2) for i in box.tolist()]
            print(
                f"Detected {model.config.id2label[label.item()]} with confidence "
                f"{round(score.item(), 3)} at location {box}"
            )


if __name__ == "__main__":
    fire.Fire(main)
