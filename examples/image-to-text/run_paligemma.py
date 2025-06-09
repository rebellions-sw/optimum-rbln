import os
import typing

import fire
from transformers import AutoProcessor

from optimum.rbln import RBLNPaliGemmaForConditionalGeneration


def main(
    model_id: str = "google/paligemma-3b-mix-448",
    batch_size: int = 1,
    from_transformers: bool = False,
    prompt: typing.Optional[str] = None,
    max_seq_len: typing.Optional[int] = None,
    tensor_parallel_size: typing.Optional[int] = 1,
    num_text_only: typing.Optional[int] = None,
):
    if from_transformers:
        processor = AutoProcessor.from_pretrained(model_id)

        model = RBLNPaliGemmaForConditionalGeneration.from_pretrained(
            model_id,
            export=True,
            rbln_config={
                "language_model": {
                    "tensor_parallel_size": tensor_parallel_size,
                    "max_seq_len": max_seq_len,
                    "use_inputs_embeds": True,
                    "batch_size": batch_size,
                    "use_attention_mask": True,
                    "prefill_chunk_size": 4096,
                },
            },
        )
        model.save_pretrained(os.path.basename(model_id))
    else:
        processor = AutoProcessor.from_pretrained(os.path.basename(model_id))
        model = RBLNPaliGemmaForConditionalGeneration.from_pretrained(
            os.path.basename(model_id),
            export=False,
        )

    # datasets = load_dataset("lmms-lab/llava-bench-in-the-wild", split="train")

    # image = datasets[0].get("image")
    # image.save("batch_0.png", "png")
    # text = datasets[0].get("question")
    # print(text)

    import requests
    from PIL import Image

    url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg?download=true"
    image = Image.open(requests.get(url, stream=True).raw)

    prompt = "caption es"
    model_inputs = processor(text=prompt, images=image, return_tensors="pt")
    generation = model.generate(**model_inputs, max_new_tokens=100, do_sample=False)
    decoded = processor.decode(generation[0], skip_special_tokens=True)
    print(decoded)


if __name__ == "__main__":
    fire.Fire(main)
