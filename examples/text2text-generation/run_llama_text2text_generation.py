# run_llama_text2text_generation.py --from_transformers --model_id neuralmagic/Meta-Llama-3-8B-Instruct-FP8-KV --n_layer 1 --kvcache_quantize
# run_llama_text2text_generation.py --from_transformers --model_id neuralmagic/Meta-Llama-3-8B-Instruct-FP8 --n_layer 1

import os
import typing

import fire
from transformers import AutoTokenizer

from optimum.rbln import RBLNLlamaForCausalLM


def main(
    model_id: str = "meta-llama/Llama-2-7b-chat-hf",
    batch_size: int = 1,
    from_transformers: bool = False,
    tensor_parallel_size: typing.Optional[int] = 4,
    host_run: bool = False,
    n_layer: int = None,
    kvcache_quantize: bool = False,
):
    # Example input sentences for the model
    sentences = [
        [{"role": "user", "content": "Name the largest country in the world?"}],
        [{"role": "user", "content": "What is Artificial intelligence?"}],
    ]

    # from transformers import AutoConfig, AutoModelForCausalLM

    # # quantization_config가 있으면 hf에서 인식하려고 해서 실패함
    # cfg = AutoConfig.from_pretrained(model_id)
    # if hasattr(cfg, "quantization_config"):
    #     del cfg.quantization_config

    # # 아래는 model weight 없을때 다운받는용 (지금 safetensor load가 directory 기반으로만 동작하기때문)
    # xx = AutoModelForCausalLM.from_pretrained(model_id, config=cfg)
    # return
    hf_kwargs = {} if n_layer is None else {"num_hidden_layers": n_layer}

    if host_run:
        model = RBLNLlamaForCausalLM.get_quantized_model(
            model_id=model_id,
            **hf_kwargs,
            rbln_quantization={
                "format": "rbln",
                "precision": "fp8_exp",
                "kvcache": "fp8" if kvcache_quantize else "fp16",
            },
        )
    elif from_transformers:
        model = RBLNLlamaForCausalLM.from_pretrained(
            model_id=model_id,
            export=True,
            **hf_kwargs,
            # The following arguments are specific to RBLN compilation
            rbln_batch_size=batch_size,
            rbln_tensor_parallel_size=tensor_parallel_size,
            rbln_quantization={
                "format": "rbln",
                "precision": "fp8_exp",
                "kvcache": "fp8" if kvcache_quantize else "fp16",
            },
        )
        model.save_pretrained(os.path.basename(model_id))
    else:
        # Load compiled model
        model = RBLNLlamaForCausalLM.from_pretrained(model_id=os.path.basename(model_id), export=False)

    # Prepare inputs
    sentences = [sentences[i % 2] for i in range(batch_size)]
    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    texts = [
        tokenizer.apply_chat_template(sentence, add_generation_prompt=True, tokenize=False) for sentence in sentences
    ]
    inputs = tokenizer(texts, return_tensors="pt", padding=True)

    # Generate
    output_sequence = model.generate(
        **inputs,
        do_sample=False,
        max_new_tokens=16,
    )

    # Decode and print the model's responses
    for i in range(batch_size):
        generated_texts = tokenizer.decode(
            output_sequence[i], skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        print("\033[32m" + f"batch {i} : " + "\033[0m\n" + generated_texts)


if __name__ == "__main__":
    fire.Fire(main)
