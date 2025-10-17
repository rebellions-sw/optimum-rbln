import os
import typing

import torch
import fire
from transformers import AutoTokenizer, GptOssForCausalLM

from optimum.rbln import RBLNGptOssForCausalLM


# def eager_attention_forward(
#     module: nn.Module,
#     query: torch.Tensor,
#     key: torch.Tensor,
#     value: torch.Tensor,
#     attention_mask: Optional[torch.Tensor],
#     scaling: float,
#     dropout: float = 0.0,
#     **kwargs,
# ):
#     key_states = repeat_kv(key, module.num_key_value_groups)
#     value_states = repeat_kv(value, module.num_key_value_groups)
#     attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
#     if attention_mask is not None:
#         causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
#         attn_weights = attn_weights + causal_mask

#     # sinks = module.sinks.reshape(1, -1, 1, 1).expand(query.shape[0], -1, query.shape[-2], -1)
#     # combined_logits = torch.cat([attn_weights, sinks], dim=-1)

#     # This was not in the original implementation and slightly affect results; it prevents overflow in BF16/FP16
#     # when training with bsz>1 we clamp max values.

#     # combined_logits = combined_logits - combined_logits.max(dim=-1, keepdim=True).values
#     probs = F.softmax(attn_weights, dim=-1, dtype=attn_weights.dtype)
#     # scores = probs[..., :-1]  # we drop the sink here
#     attn_weights = nn.functional.dropout(probs, p=dropout, training=module.training)
#     attn_output = torch.matmul(attn_weights, value_states)
#     attn_output = attn_output.transpose(1, 2).contiguous()
#     return attn_output, attn_weights


def main(
    model_id: str = "openai/gpt-oss-20b",
    batch_size: int = 1,
    from_transformers: bool = False,
    max_seq_len: typing.Optional[int] = 4096,
    tensor_parallel_size: typing.Optional[int] = 1,
    kvcache_partition_len: typing.Optional[int] = None,
    diff: bool = False,
    n_layers: int = 2,
):
    if from_transformers:
        model = RBLNGptOssForCausalLM.from_pretrained(
            model_id,
            export=True,
            rbln_batch_size=batch_size,
            rbln_max_seq_len=max_seq_len,
            rbln_tensor_parallel_size=tensor_parallel_size,
            rbln_kvcache_partition_len=kvcache_partition_len,
            num_hidden_layers=n_layers,
            dtype=torch.bfloat16,
        )
        model.save_pretrained(os.path.basename(model_id))
    else:
        model = RBLNGptOssForCausalLM.from_pretrained(os.path.basename(model_id), export=False)

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

    output_sequence = rbln_outputs.sequences
    logits = rbln_outputs.logits

    if diff:
        golden_model = GptOssForCausalLM.from_pretrained(
            model_id,
            num_hidden_layers=n_layers,
            _attn_implementation="eager",
        )
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

    if diff:
        from scipy import stats

        for i, (r, g) in enumerate(zip(logits, golden_logits)):
            print(
                f"step {i} : {stats.pearsonr(r.detach().numpy().reshape(-1), g.detach().numpy().reshape(-1)).statistic}"
            )

    breakpoint()


if __name__ == "__main__":
    fire.Fire(main)
