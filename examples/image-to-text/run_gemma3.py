from typing import Optional
import os
import json

import fire
from datasets import load_dataset
from transformers import AutoProcessor, Gemma3Config, Gemma3ForConditionalGeneration
from optimum.rbln import RBLNGemma3ForConditionalGeneration


model_id = "google/gemma-3-4b-it"
processor = AutoProcessor.from_pretrained(model_id, padding_side="left")


def get_inputs(batch_size):
    dataset = load_dataset("lmms-lab/llava-bench-in-the-wild", split="train").shuffle(seed=42)
    messages = [
        [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful assistant."}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": dataset[i]["image"]
                    },
                    {"type": "text", "text": dataset[i]["question"]},
                ],
            },
        ]
        for i in range(batch_size)
    ]
    images = [[dataset[i]["image"]] for i in range(batch_size)]

    text = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )

    inputs = processor(text=text, images=images, return_tensors="pt", padding=True)

    return inputs


def main(
    compile: bool = False,
    diff: bool = False,
    batch_size: int = 2,
    kv_partition_len: Optional[int] = None,
    tensor_parallel_size: int = 1,
    n_layers: Optional[int] = None,
):
    inputs = get_inputs(batch_size)

    input_len = inputs["input_ids"].shape[-1]
    hf_kwargs ={}
    if n_layers is not None:
        hf_config = Gemma3Config.from_pretrained(model_id)
        text_config = json.loads(hf_config.text_config.to_json_string())
        text_config["num_hidden_layers"] = n_layers
        hf_kwargs = {"text_config": text_config}


    if compile:
        kwargs = {}
        rbln_config={
            "language_model": {
                "use_attention_mask": True,
                "max_seq_len": 32768,
                # "kvcache_partition_len": kv_partition_len,
                "batch_size": batch_size,
                "tensor_parallel_size": tensor_parallel_size,
                "use_inputs_embeds": True,
            }
        }
        
        if kv_partition_len is not None:
            rbln_config["language_model"].update({"kvcache_partition_len": kv_partition_len})

        rbln_model = RBLNGemma3ForConditionalGeneration.from_pretrained(
            model_id,
            export=True,
            rbln_config=rbln_config,
            **hf_kwargs,
            # config=hf_config,
            # TODO RBLNGemma3ForConditionalGeneration의 batch_size가 CausalLM으로 넘겨받는 식이 되어야하는가? X
        )
        rbln_model.save_pretrained(os.path.basename(model_id) + f"_b{batch_size}")
    else:
        rbln_model = RBLNGemma3ForConditionalGeneration.from_pretrained(
            os.path.basename(model_id) + f"_b{batch_size}",
            export=False,
        )

    if diff:
        model = Gemma3ForConditionalGeneration.from_pretrained(model_id,**hf_kwargs).eval()

        output = rbln_model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
            return_dict_in_generate=True,
            output_logits=True,
        )
        rbln_generated_ids = output.sequences[:, input_len:]
        rbln_logits = output.logits

        decoded = processor.batch_decode(rbln_generated_ids, skip_special_tokens=True)
        print("--RBLN Result--")
        print(decoded)
        # breakpoint()

        output = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
            return_dict_in_generate=True,
            output_logits=True,
        )
        golden_generated_ids = output.sequences[:, input_len:]
        golden_logits = output.logits

        decoded = processor.batch_decode(golden_generated_ids, skip_special_tokens=True)
        print("--Golden Result--")
        print(decoded)

        # from scipy import stats

        # pearsonr = stats.pearsonr(
        #     rbln_logits[0].numpy().reshape(-1),
        #     golden_logits[0].numpy().reshape(-1),
        # )
        # print("prefill pearsonr")
        # print(pearsonr.statistic)
        # print("decoder pearsonr")
        # # breakpoint()
        # pearsonr = stats.pearsonr(
        #     rbln_logits[10].numpy().reshape(-1),
        #     golden_logits[10].numpy().reshape(-1),
        # )
        # print(pearsonr.statistic)


# breakpoint()
if __name__ == "__main__":
    fire.Fire(main)
