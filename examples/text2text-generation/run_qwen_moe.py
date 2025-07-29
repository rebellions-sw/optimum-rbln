import os
import typing

import fire
import torch
from torch import nn
from transformers import AutoTokenizer, Qwen2MoeForCausalLM
from transformers.activations import ACT2FN

from optimum.rbln import RBLNQwen2MoeForCausalLM


class CustomQwen2MoeMLP(nn.Module):
    def __init__(self, expert_list):
        super().__init__()
        self.config = expert_list[0].config
        self.hidden_size = expert_list[0].hidden_size
        self.intermediate_size = expert_list[0].intermediate_size
        self.act_fn = ACT2FN[self.config.hidden_act]

        # RBLN-optimized MLP
        self.num_experts = len(expert_list)
        self.gate_proj = nn.Linear(self.hidden_size, self.num_experts * self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.num_experts * self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.num_experts * self.intermediate_size, self.hidden_size, bias=False)
        self.gate_proj.weight.data = torch.cat([expert.gate_proj.weight.data for expert in expert_list], dim=0)
        self.up_proj.weight.data = torch.cat([expert.up_proj.weight.data for expert in expert_list], dim=0)
        self.down_proj.weight.data = torch.cat([expert.down_proj.weight.data for expert in expert_list], dim=1)

    def forward(self, x, masked_routing_weights):
        # masked_routing_weights: (batch * sequence_length, num_experts)
        # x: (batch * sequence_length, hidden_size)
        # y: (batch * sequence_length, num_experts * intermediate_size)
        y = self.act_fn(self.gate_proj(x)) * self.up_proj(x)

        # elementwise multiplication of y and routing_weights
        y = y.reshape(-1, self.num_experts, self.intermediate_size) * masked_routing_weights[:, :, None]
        y = y.reshape(-1, self.num_experts * self.intermediate_size)

        return self.down_proj(y)


class CustomQwen2MoeBlock(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.num_experts = model.num_experts
        self.top_k = model.top_k
        self.num_topk_prob = model.norm_topk_prob
        self.gate = model.gate
        self.shared_expert = model.shared_expert
        self.shared_expert_gate = model.shared_expert_gate
        self.experts = CustomQwen2MoeMLP(model.experts)

    def get_masked_routing_weights(self, router_logits):
        # routing_weights: (batch * sequence_length, n_experts)
        routing_weights = torch.nn.functional.softmax(router_logits, dim=1, dtype=torch.float)

        # selected_experts: (batch * sequence_length, top_k)
        _, selected_experts = torch.topk(routing_weights, k=self.top_k, dim=-1)
        mask = torch.zeros_like(routing_weights)
        mask = mask.scatter(1, selected_experts, 1.0)

        masked_routing_weights = routing_weights * mask

        return masked_routing_weights

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """ """
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)
        masked_routing_weights = self.get_masked_routing_weights(router_logits)
        final_hidden_states = self.experts(hidden_states, masked_routing_weights)

        shared_expert_output = self.shared_expert(hidden_states)
        shared_expert_output = (
            torch.nn.functional.sigmoid(self.shared_expert_gate(hidden_states)) * shared_expert_output
        )

        final_hidden_states = final_hidden_states + shared_expert_output

        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits


def replace_qwen2moe_block(model: Qwen2MoeForCausalLM):
    for layer in model.model.layers:
        layer.mlp = CustomQwen2MoeBlock(layer.mlp)


def main(
    model_id: str = "Qwen/Qwen1.5-MoE-A2.7B-Chat",
    batch_size: int = 1,
    from_transformers: bool = False,
    max_seq_len: typing.Optional[int] = None,
    tensor_parallel_size: typing.Optional[int] = 1,
    kvcache_partition_len: typing.Optional[int] = None,
    diff: bool = False,
):
    if from_transformers:
        model = RBLNQwen2MoeForCausalLM.from_pretrained(
            model_id,
            export=True,
            rbln_batch_size=batch_size,
            rbln_max_seq_len=max_seq_len,
            rbln_tensor_parallel_size=tensor_parallel_size,
            rbln_kvcache_partition_len=kvcache_partition_len,
            num_hidden_layers=1,
        )
        model.save_pretrained(os.path.basename(model_id))
    else:
        model = RBLNQwen2MoeForCausalLM.from_pretrained(os.path.basename(model_id), export=False)

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
    output_sequence = model.generate(
        **inputs,
        do_sample=False,
        max_new_tokens=64,
    )

    # Decode and print the model's responses
    for i in range(batch_size):
        generated_texts = tokenizer.decode(
            output_sequence[i], skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        print("\033[32m" + f"batch {i} : " + "\033[0m\n" + generated_texts)


if __name__ == "__main__":
    fire.Fire(main)
