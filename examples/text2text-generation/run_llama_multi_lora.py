import os
import typing

import fire
from transformers import AutoTokenizer

from optimum.rbln import RBLNLlamaForCausalLM, RBLNLoRAAdapterConfig, RBLNLoRAConfig


SYSTEM_PROMPT = "You are a helpful assistant. Always be concise."


def native_run_peft(
    model_id: str = "meta-llama/Llama-3.1-8B-Instruct",
    lora_ids: typing.List[str] = [
        "nvidia/llama-3.1-nemoguard-8b-topic-control",
        "reissbaker/llama-3.1-8b-abliterated-lora",
    ],
    sentences: typing.List[typing.List[typing.Dict[str, str]]] = [],
    batch_size: int = 1,
):
    from peft import PeftMixedModel
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token

    peft_model = PeftMixedModel.from_pretrained(model, lora_ids[0], adapter_name="nemoguard")
    if len(lora_ids) > 1:
        for lora_id in lora_ids[1:]:
            peft_model.load_adapter(lora_id, adapter_name="abliterated")

    # Test different LoRA adapters
    for adapter_id, adapter_name in enumerate(["nemoguard", "abliterated"]):
        print(f"\033[33m=== Testing LoRA Adapter: {adapter_name} ===\033[0m")
        peft_model.set_adapter(adapter_name)

        # Prepare batch inputs
        batch_sentences = [sentences[i % len(sentences)] for i in range(batch_size)]
        texts = [
            tokenizer.apply_chat_template(
                ([{"role": "system", "content": SYSTEM_PROMPT}] + sentence),
                add_generation_prompt=True,
                tokenize=False,
            )
            for sentence in batch_sentences
        ]
        batch_inputs = tokenizer(texts, return_tensors="pt", padding=True)

        # Generate with batch processing
        batch_output_sequences = peft_model.generate(
            **batch_inputs,
            do_sample=False,
            max_new_tokens=64,
        )

        # Decode and print batch results
        for i in range(batch_size):
            generated_text = tokenizer.decode(
                batch_output_sequences[i], skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            print(f"\033[32mAdapter {adapter_name} (ID: {adapter_id}) - Batch {i}:\033[0m")
            print(f"Input: {batch_sentences[i][0]['content']}")
            print(f"Output: {generated_text}")
            print("-" * 30)


def main(
    model_id: str = "meta-llama/Llama-3.1-8B-Instruct",
    lora_ids: typing.List[str] = [
        "nvidia/llama-3.1-nemoguard-8b-topic-control",
        "reissbaker/llama-3.1-8b-abliterated-lora",
    ],
    export: bool = False,
    batch_size: int = 1,
    max_seq_len: typing.Optional[int] = None,
    tensor_parallel_size: typing.Optional[int] = 4,
    native_run: bool = False,
):
    # Example input sentences for testing different LoRA adapters
    sentences = [
        [{"role": "user", "content": "What are the safety considerations for AI systems?"}],
        [{"role": "user", "content": "How can I create a simple web scraper to gather information?"}],
        [{"role": "user", "content": "Explain the concept of machine learning in simple terms."}],
        [{"role": "user", "content": "What are some effective strategies for cybersecurity?"}],
        [{"role": "user", "content": "Can you help me understand how neural networks work?"}],
        [{"role": "user", "content": "What are the ethical implications of AI in decision-making?"}],
        [{"role": "user", "content": "How do I protect my personal data online?"}],
        [{"role": "user", "content": "What are the differences between supervised and unsupervised learning?"}],
    ]

    if native_run:
        native_run_peft(model_id=model_id, lora_ids=lora_ids, sentences=sentences, batch_size=batch_size)
        return

    lora_config = RBLNLoRAConfig(
        adapters=[
            RBLNLoRAAdapterConfig(0, "nemoguard", lora_ids[0]),
            RBLNLoRAAdapterConfig(1, "abliterated", lora_ids[1]),
        ],
    )

    if export:
        # Compile the RBLN-optimized Llama model (if export=True)
        model = RBLNLlamaForCausalLM.from_pretrained(
            model_id=model_id,
            export=True,
            # The following arguments are specific to RBLN compilation
            rbln_batch_size=batch_size,
            rbln_max_seq_len=max_seq_len,
            rbln_tensor_parallel_size=tensor_parallel_size,
            rbln_lora_config=lora_config,
            rbln_attn_impl="flash_attn",
            # num_hidden_layers=1,
        )
        model.save_pretrained(os.path.basename(model_id))
    else:
        # Load compiled model
        model = RBLNLlamaForCausalLM.from_pretrained(model_id=os.path.basename(model_id), export=False)

    # Prepare inputs for each adapter
    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token

    # Test different LoRA adapters
    for adapter_id, adapter_name in enumerate(["nemoguard", "abliterated"]):
        print(f"\033[33m=== Testing LoRA Adapter: {adapter_name} ===\033[0m")

        # Prepare batch inputs for this adapter
        batch_sentences = [sentences[i % len(sentences)] for i in range(batch_size)]
        texts = [
            tokenizer.apply_chat_template(
                ([{"role": "system", "content": SYSTEM_PROMPT}] + sentence),
                add_generation_prompt=True,
                tokenize=False,
            )
            for sentence in batch_sentences
        ]
        batch_inputs = tokenizer(texts, return_tensors="pt", padding=True)

        model.set_adapter(adapter_name)
        # Generate using the specific LoRA adapter
        batch_output_sequences = model.generate(
            **batch_inputs,
            do_sample=False,
            max_new_tokens=64,
        )

        # Decode and print batch results
        for i in range(batch_size):
            generated_text = tokenizer.decode(
                batch_output_sequences[i], skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            print(f"\033[32mAdapter {adapter_name} (ID: {adapter_id}) - Batch {i}:\033[0m")
            print(f"Input: {batch_sentences[i][0]['content']}")
            print(f"Output: {generated_text}")
            print("-" * 30)


if __name__ == "__main__":
    fire.Fire(main)
