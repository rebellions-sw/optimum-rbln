import os

import fire
import requests
import torch
from PIL import Image
from transformers import AutoProcessor, GroundingDinoForObjectDetection

from optimum.rbln.transformers import RBLNGroundingDinoForObjectDetection


def main(compile: bool = False, native_run: bool = False, rbln_run: bool = False):
    model_id = "IDEA-Research/grounding-dino-tiny"

    processor = AutoProcessor.from_pretrained(model_id, max_length=256)
    if native_run:
        model = GroundingDinoForObjectDetection.from_pretrained(model_id)

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    url2 = "http://images.cocodataset.org/val2017/000000000139.jpg"
    image = Image.open(requests.get(url, stream=True).raw).resize((704, 1280))
    image2 = Image.open(requests.get(url2, stream=True).raw)

    text = "a cat. a remote control."
    longest_edge = processor.image_processor.size["longest_edge"]

    inputs = processor(
        images=image,
        text=text,
        padding="max_length",
        do_pad=True,
        pad_size={"height": longest_edge, "width": longest_edge},
        return_tensors="pt",
    )

    if compile:
        rbln_model = RBLNGroundingDinoForObjectDetection.from_pretrained(
            model_id,
            export=True,
            model_save_dir=os.path.basename(model_id),
            rbln_config={
                "text_backbone": {
                    "model_input_names": ["input_ids", "attention_mask", "token_type_ids", "position_ids"],
                    "input_shapes": [(1, 256), (1, 256, 256), (1, 256), (1, 256)],
                },
            },
        )

    if rbln_run:
        if not compile:
            rbln_model = RBLNGroundingDinoForObjectDetection.from_pretrained(os.path.basename(model_id), export=False)

        with torch.inference_mode():
            rbln_outputs = rbln_model(**inputs)

        rbln_results = processor.post_process_grounded_object_detection(
            rbln_outputs,
            inputs.input_ids,
            box_threshold=0.4,
            text_threshold=0.3,
            target_sizes=[image.size[::-1]],
        )
        print("RBLN Model Outputs:")
        print(rbln_outputs.last_hidden_state)
        print("RBLN Results:")
        print(rbln_results)

    if native_run:
        with torch.inference_mode():
            golden_outputs = model(**inputs)

        golden_results = processor.post_process_grounded_object_detection(
            golden_outputs,
            inputs.input_ids,
            box_threshold=0.4,
            text_threshold=0.3,
            target_sizes=[image.size[::-1]],
        )
        print("CPU Model Outputs:")
        print(golden_outputs.last_hidden_state)
        print("CPU Results:")
        print(golden_results)

    breakpoint()


if __name__ == "__main__":
    fire.Fire(main)
