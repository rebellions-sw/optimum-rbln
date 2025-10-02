import os
import typing

import fire
from transformers import AutoModelForCausalLM, AutoTokenizer

from optimum.rbln import RBLNQwen3MoeForCausalLM


def main(
    model_id: str = "Qwen/Qwen3-30B-A3B-Thinking-2507",
    batch_size: int = 64,
    from_transformers: bool = False,
    max_seq_len: typing.Optional[int] = 4096,
    tensor_parallel_size: typing.Optional[int] = 1,
    kvcache_partition_len: typing.Optional[int] = None,
    diff: bool = False,
    n_layers: int = 1
):
    if from_transformers:
        model = RBLNQwen3MoeForCausalLM.from_pretrained(
            model_id,
            export=True,
            rbln_batch_size=batch_size,
            rbln_max_seq_len=max_seq_len,
            rbln_tensor_parallel_size=tensor_parallel_size,
            rbln_kvcache_partition_len=kvcache_partition_len,
            num_hidden_layers=n_layers,
        )
        model.save_pretrained(os.path.basename(model_id))
    else:
        model = RBLNQwen3MoeForCausalLM.from_pretrained(os.path.basename(model_id), export=False)

    # model = Qwen2MoeForCausalLM.from_pretrained(model_id, num_hidden_layers=1)
    # replace_qwen2moe_block(model)

    # Example input sentences for the model
    sentences = [
        [{"role": "user", "content": "Name the largest country in the world?"}],
        [{"role": "user", "content": "What is Artificial intelligence?"}],
    ]

    # Prepare inputs
    sentences = [sentences[i % 2] for i in range(batch_size)]
    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    texts = [
        tokenizer.apply_chat_template(sentence, add_generation_prompt=True, tokenize=False) for sentence in sentences
    ]
    inputs = tokenizer(texts, return_tensors="pt", padding=True)

    # Generate
    rbln_outputs = model.generate(
        **inputs,
        do_sample=False,
        max_new_tokens=4,
        return_dict_in_generate=True,
        output_logits=True,
    )
    rbln_outputs = model.generate(
        **inputs,
        do_sample=False,
        max_new_tokens=4,
        return_dict_in_generate=True,
        output_logits=True,
    )

    output_sequence = rbln_outputs.sequences
    logits = rbln_outputs.logits

    if diff:
        golden_model = AutoModelForCausalLM.from_pretrained(model_id, num_hidden_layers=n_layers)
        golden_outputs = golden_model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=4,
            return_dict_in_generate=True,
            output_logits=True,
        )
        golden_logits = golden_outputs.logits
        print("Golden Sequence")
        for i in range(batch_size):
            generated_texts = tokenizer.decode(
                golden_outputs.sequences[i], skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            print("\033[32m" + f"batch {i} : " + "\033[0m\n" + generated_texts)

    print("RBLN Sequence")
    # Decode and print the model's responses
    for i in range(batch_size):
        generated_texts = tokenizer.decode(
            output_sequence[i], skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        print("\033[32m" + f"batch {i} : " + "\033[0m\n" + generated_texts)

    breakpoint()

    if diff:
        from scipy import stats
        for i, (r, g) in enumerate(zip(logits, golden_logits)):
            print(
                f"step {i} : {stats.pearsonr(r.detach().numpy().reshape(-1), g.detach().numpy().reshape(-1)).statistic}"
            )


if __name__ == "__main__":
    fire.Fire(main)
