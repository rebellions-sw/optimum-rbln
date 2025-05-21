import json
import os
from typing import Optional

import fire
from transformers import AutoTokenizer, Gemma3ForCausalLM

from optimum.rbln import RBLNGemma3ForCausalLM


model_id = "google/gemma-3-1b-it"
tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")


def get_inputs(batch_size):
    dataset_path = "/mnt/shared_data/.cache/dataset/BookCorpus2/bookcorpus_texts.json"

    with open(dataset_path, "r", encoding="utf-8") as f:
        restored_texts = json.load(f)

    messages = [
        [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a helpful assistant. Summarize the following text.",
                    }
                ],
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": restored_texts[-i][:8192]}],
            },
        ]
        for i in range(batch_size)
    ]

    texts = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    inputs = tokenizer(texts, padding=True, return_tensors="pt")

    return inputs


def main(
    compile: bool = False,
    diff: bool = False,
    batch_size: int = 2,
    kv_partition_len: Optional[int] = None,
    tensor_parallel_size: int = 1,
    n_layers: Optional[int] = None,
    sliding_window_pattern: Optional[int] = None,
    use_inputs_embeds: bool = False,
    use_attention_mask: bool = False,
):
    inputs = get_inputs(batch_size)

    input_len = inputs["input_ids"].shape[-1]

    hf_kwargs = {}
    if n_layers is not None:
        hf_kwargs.update({"num_hidden_layers": n_layers})
    if sliding_window_pattern is not None:
        hf_kwargs.update({"sliding_window_pattern": sliding_window_pattern})

    if compile:
        kwargs = {}
        if kv_partition_len is not None:
            kwargs.update({"rbln_kvcache_partition_len": kv_partition_len})

        rbln_model = RBLNGemma3ForCausalLM.from_pretrained(
            model_id,
            export=True,
            rbln_max_seq_len=32768,
            rbln_batch_size=batch_size,
            rbln_use_attention_mask=use_attention_mask,
            rbln_tensor_parallel_size=tensor_parallel_size,
            rbln_use_inputs_embeds=use_inputs_embeds,
            **hf_kwargs,
            **kwargs,
        )
        rbln_model.save_pretrained(os.path.basename(model_id) + f"_b{batch_size}")
    else:
        rbln_model = RBLNGemma3ForCausalLM.from_pretrained(
            os.path.basename(model_id) + f"_b{batch_size}",
            export=False,
        )

    if diff:
        model = Gemma3ForCausalLM.from_pretrained(model_id, **hf_kwargs).eval()

        output = rbln_model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
            return_dict_in_generate=True,
            output_logits=True,
        )
        rbln_generated_ids = output.sequences[:, input_len:]
        rbln_logits = output.logits

        decoded = tokenizer.batch_decode(rbln_generated_ids, skip_special_tokens=True)
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

        decoded = tokenizer.batch_decode(golden_generated_ids, skip_special_tokens=True)
        print("--Golden Result--")
        print(decoded)

        from scipy import stats

        pearsonr = stats.pearsonr(
            rbln_logits[0].numpy().reshape(-1),
            golden_logits[0].numpy().reshape(-1),
        )
        print("prefill pearsonr")
        print(pearsonr.statistic)
        print("decoder pearsonr")
        # breakpoint()
        pearsonr = stats.pearsonr(
            rbln_logits[10].numpy().reshape(-1),
            golden_logits[10].numpy().reshape(-1),
        )
        print(pearsonr.statistic)


# breakpoint()
if __name__ == "__main__":
    fire.Fire(main)
