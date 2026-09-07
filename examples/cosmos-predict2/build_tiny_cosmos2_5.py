"""Build a tiny-random Cosmos-Predict2.5 diffusers pipeline for optimum-rbln pytest.

Mirrors the real nvidia/Cosmos-Predict2.5-2B (diffusers/base/post-trained) structure:
- transformer: real config shrunk (2 layers, 2 heads x head_dim 128, crossattn projection kept:
  crossattn_proj_in_channels = TE num_hidden_layers x TE hidden = 2 x 128 = 256)
- vae: REAL Wan2.1 VAE config verbatim (random weights, bf16) — the RBLN wrapper's cache
  shapes are derived from this exact architecture, so it cannot be shrunk
- text_encoder: tiny Qwen2.5-VL (hidden 128, head_dim 64 to satisfy RBLN attention kernels)
- tokenizer / scheduler: copied verbatim from the real snapshot
- safety_checker: saved as (None, None) — tests pass a mock at load time
"""

import json
import shutil
from pathlib import Path

import torch
from diffusers import (
    AutoencoderKLWan,
    Cosmos2_5_PredictBasePipeline,
    CosmosTransformer3DModel,
    UniPCMultistepScheduler,
)
from transformers import AutoTokenizer, Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration


SNAP = Path(
    "/mnt/shared_data/groups/sw_dev/.cache/huggingface/hub/models--nvidia--Cosmos-Predict2.5-2B"
    "/snapshots/0d37c7498f54cee3c599d438d895a0a4a8608064"
)
OUT = Path.home() / "tiny-random-cosmos-2.5-predict"

torch.manual_seed(0)

# --- text encoder: tiny Qwen2.5-VL, RBLN-safe dims (head_dim 64) ---
TE_HIDDEN, TE_LAYERS = 128, 2
te_config = Qwen2_5_VLConfig(
    text_config={
        "hidden_size": TE_HIDDEN,
        "intermediate_size": 256,
        "num_hidden_layers": TE_LAYERS,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "rope_scaling": {"mrope_section": [8, 12, 12], "rope_type": "default", "type": "default"},
        "rope_theta": 1000000.0,
        "max_position_embeddings": 128000,
        "vocab_size": 152064,
    },
    vision_config={
        # structural fields mirror the real Cosmos-Reason1 vision config
        "depth": 2,
        "hidden_size": 128,
        "intermediate_size": 256,
        "num_heads": 2,
        "out_hidden_size": TE_HIDDEN,
        "fullatt_block_indexes": [1],
        "hidden_act": "silu",
        "in_channels": 3,
        "patch_size": 14,
        "spatial_merge_size": 2,
        "spatial_patch_size": 14,
        "temporal_patch_size": 2,
        "tokens_per_second": 2,
        "window_size": 112,
    },
    vocab_size=152064,
)
text_encoder = Qwen2_5_VLForConditionalGeneration(te_config)

# --- transformer: real config shrunk ---
real_tf = json.loads((SNAP / "transformer" / "config.json").read_text())
tf_config = {
    **{k: v for k, v in real_tf.items() if not k.startswith("_")},
    "num_layers": 2,
    "num_attention_heads": 2,
    "attention_head_dim": 128,
    "adaln_lora_dim": 32,
    "text_embed_dim": 128,
    "encoder_hidden_states_channels": 128,
    "crossattn_proj_in_channels": TE_LAYERS * TE_HIDDEN,  # concat of hidden_states[1:]
}
transformer = CosmosTransformer3DModel(**tf_config)

# --- vae: real architecture, random weights ---
real_vae = json.loads((SNAP / "vae" / "config.json").read_text())
vae = AutoencoderKLWan(**{k: v for k, v in real_vae.items() if not k.startswith("_")})

# --- scheduler/tokenizer: verbatim from the real snapshot ---
scheduler = UniPCMultistepScheduler.from_config(json.loads((SNAP / "scheduler" / "scheduler_config.json").read_text()))
tokenizer = AutoTokenizer.from_pretrained(SNAP / "tokenizer")

pipe = Cosmos2_5_PredictBasePipeline(
    text_encoder=text_encoder.to(torch.bfloat16),
    tokenizer=tokenizer,
    transformer=transformer.to(torch.bfloat16),
    vae=vae.to(torch.bfloat16),
    scheduler=scheduler,
    safety_checker=None,  # requires the locally patched diffusers (stock would build the real guardrail)
)
if OUT.exists():
    shutil.rmtree(OUT)
pipe.save_pretrained(OUT)

# make sure the saved index really has no safety checker
index = json.loads((OUT / "model_index.json").read_text())
assert index["safety_checker"] == [None, None], index["safety_checker"]
size = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file())
print(f"saved {OUT} ({size / 1e6:.1f} MB)")
