import os
import urllib

import fire
from PIL import Image
from transformers import ViTFeatureExtractor

from optimum.rbln import RBLNViTForImageClassification


def main(
    model_id: str = "google/vit-large-patch16-224",
    from_transformers: bool = False,
    batch_size: int = 1,
):
    img_url = "https://rbln-public.s3.ap-northeast-2.amazonaws.com/images/tabby.jpg"
    img_path = "./tabby.jpg"
    if not os.path.exists(img_path):
        with urllib.request.urlopen(img_url) as response, open(img_path, "wb") as f:
            f.write(response.read())
    image = Image.open(img_path)

    feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-large-patch16-224")
    inputs = feature_extractor(images=[image] * batch_size, return_tensors="pt")

    if from_transformers:
        model = RBLNViTForImageClassification.from_pretrained(
            model_id,
            export=True,
            rbln_batch_size=batch_size,
        )
        model.save_pretrained(os.path.basename(model_id))
    else:
        model = RBLNViTForImageClassification.from_pretrained(model_id=os.path.basename(model_id), export=False)

    logits = model(**inputs)
    labels = logits.argmax(-1)
    print("predicted label:", [model.config.id2label[label.item()] for label in labels])


if __name__ == "__main__":
    fire.Fire(main)
