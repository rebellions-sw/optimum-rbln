import os
import typing

import fire

from optimum.rbln import RBLNLlamaForCausalLM, RBLNLoRAAdapterConfig, RBLNLoRAConfig


def main(
    model_id: str = "meta-llama/Llama-3.1-8B-Instruct",
    lora_ids: typing.List[str] = [
        "nvidia/llama-3.1-nemoguard-8b-topic-control",
        "reissbaker/llama-3.1-8b-abliterated-lora",
    ],
    batch_size: int = 1,
    max_seq_len: typing.Optional[int] = None,
    tensor_parallel_size: typing.Optional[int] = 4,
):
    lora_config = RBLNLoRAConfig(
        adapters=[
            RBLNLoRAAdapterConfig(0, "nemoguard", lora_ids[0]),
            RBLNLoRAAdapterConfig(1, "abliterated", lora_ids[1]),
        ],
    )

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
        num_hidden_layers=1,
    )
    model.save_pretrained(os.path.basename(model_id))


if __name__ == "__main__":
    fire.Fire(main)
