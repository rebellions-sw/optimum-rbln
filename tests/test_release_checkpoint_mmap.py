import glob
import json
import struct
import tempfile
import unittest

import torch
from safetensors.torch import save_file
from transformers import (
    MixtralConfig,
    MixtralForCausalLM,
    Qwen2MoeConfig,
    Qwen2MoeForCausalLM,
    Qwen3MoeConfig,
    Qwen3MoeForCausalLM,
)

from optimum.rbln import RBLNMixtralForCausalLM, RBLNQwen2MoeForCausalLM, RBLNQwen3MoeForCausalLM


TINY = {
    "vocab_size": 256,
    "hidden_size": 64,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "max_position_embeddings": 64,
}
CONFIG = Qwen3MoeConfig(
    **TINY,
    intermediate_size=128,
    moe_intermediate_size=128,
    num_experts=16,
    num_experts_per_tok=2,
    decoder_sparse_step=1,
)
FAMILIES = [
    (Qwen3MoeForCausalLM, RBLNQwen3MoeForCausalLM, CONFIG, "model.layers.0.mlp.experts.0.gate_proj.weight"),
    (
        Qwen2MoeForCausalLM,
        RBLNQwen2MoeForCausalLM,
        Qwen2MoeConfig(
            **TINY,
            intermediate_size=128,
            moe_intermediate_size=128,
            shared_expert_intermediate_size=128,
            num_experts=16,
            num_experts_per_tok=2,
            decoder_sparse_step=1,
        ),
        "model.layers.0.mlp.experts.0.gate_proj.weight",
    ),
    (
        MixtralForCausalLM,
        RBLNMixtralForCausalLM,
        MixtralConfig(**TINY, intermediate_size=128, num_local_experts=16, num_experts_per_tok=2),
        "model.layers.0.block_sparse_moe.experts.0.w1.weight",
    ),
]


def _checkpoint_ranges():
    ranges = []
    for line in open("/proc/self/maps"):
        parts = line.split()
        if len(parts) >= 6 and (parts[5].endswith(".safetensors") or "/blobs/" in parts[5]):
            start, end = parts[0].split("-")
            ranges.append((int(start, 16), int(end, 16)))
    return ranges


def _file_backed(model):
    ranges = _checkpoint_ranges()
    return [
        name
        for name, t in list(model.named_parameters()) + list(model.named_buffers())
        if any(s <= t.untyped_storage().data_ptr() < e for s, e in ranges)
    ]


def _safetensors_keys(directory):
    path = glob.glob(f"{directory}/*.safetensors")[0]
    header_len = struct.unpack("<Q", open(path, "rb").read(8))[0]
    return json.loads(open(path, "rb").read()[8 : 8 + header_len]).keys()


class TestReleaseCheckpointMmap(unittest.TestCase):
    def test_per_expert_checkpoint_is_unmapped(self):
        # save_pretrained writes the per-expert layout; loading stacks the experts into anonymous
        # memory, so the remaining file-backed views must be cloned and the shard unmapped.
        for hf_cls, rbln_cls, config, expert_key in FAMILIES:
            with self.subTest(hf_cls.__name__), tempfile.TemporaryDirectory() as tmp:
                src = hf_cls(config).eval()
                src.save_pretrained(tmp)
                self.assertIn(expert_key, _safetensors_keys(tmp))

                model = rbln_cls.get_pytorch_model(tmp)
                self.assertEqual(_file_backed(model), [])
                for (name, p), (_, q) in zip(model.named_parameters(), src.named_parameters(), strict=True):
                    self.assertTrue(torch.equal(p, q), name)

    def test_fused_checkpoint_loads_equal(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Qwen3MoeForCausalLM(CONFIG).eval()
            save_file({k: v.contiguous() for k, v in src.state_dict().items()}, f"{tmp}/model.safetensors")
            src.config.save_pretrained(tmp)
            self.assertIn("model.layers.0.mlp.experts.gate_up_proj", _safetensors_keys(tmp))

            model = RBLNQwen3MoeForCausalLM.get_pytorch_model(tmp)
            for (name, p), (_, q) in zip(model.named_parameters(), src.named_parameters(), strict=True):
                self.assertTrue(torch.equal(p, q), name)


if __name__ == "__main__":
    unittest.main()
