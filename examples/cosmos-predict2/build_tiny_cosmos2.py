"""Build tiny-random Cosmos-Predict2 (t2i / v2w) diffusers pipelines for optimum-rbln pytest.

Components are saved individually and model_index.json is written by hand, because the stock
Cosmos2TextToImagePipeline.__init__ auto-builds the real guardrail when safety_checker is None.

Mirrors the real nvidia/Cosmos-Predict2-2B-* repos:
- transformer: real config shrunk (2 layers, 2 heads x head_dim 128, text_embed_dim = T5 d_model)
- vae: REAL Wan2.1 VAE config verbatim (random weights, bf16) — RBLN cache shapes require it
- text_encoder: real T5-11B config shrunk (d_model 64)
- tokenizer / scheduler / model_index: copied from the real t2i snapshot
"""

import json
import shutil
from pathlib import Path

import torch
from diffusers import AutoencoderKLWan, CosmosTransformer3DModel
from transformers import T5Config, T5EncoderModel


T2I_SNAP = Path(
    "/mnt/shared_data/groups/sw_dev/.cache/huggingface/hub/models--nvidia--Cosmos-Predict2-2B-Text2Image"
    "/snapshots/acdb5fde992a73ef0355f287977d002cbfd127e0"
)

torch.manual_seed(0)
TE_DIM = 64

real_tf = {
    k: v
    for k, v in json.loads((T2I_SNAP / "transformer" / "config.json").read_text()).items()
    if not k.startswith("_")
}
real_vae = {
    k: v for k, v in json.loads((T2I_SNAP / "vae" / "config.json").read_text()).items() if not k.startswith("_")
}
real_te = json.loads((T2I_SNAP / "text_encoder" / "config.json").read_text())
model_index = json.loads((T2I_SNAP / "model_index.json").read_text())

te_config = T5Config.from_dict(
    {
        **real_te,
        "d_model": TE_DIM,
        "d_kv": 32,
        "d_ff": 128,
        "num_heads": 2,
        "num_layers": 2,
    }
)

VARIANTS = {
    "tiny-random-cosmos2-text2image": {
        "class_name": "Cosmos2TextToImagePipeline",
        "tf_overrides": {},  # t2i: in_channels 16, rope_scale (1, 4, 4) already in the real config
    },
    "tiny-random-cosmos2-video2world": {
        "class_name": "Cosmos2VideoToWorldPipeline",
        "tf_overrides": {"in_channels": 17, "rope_scale": [1.0, 3.0, 3.0]},
    },
}

for name, spec in VARIANTS.items():
    out = Path.home() / name
    if out.exists():
        shutil.rmtree(out)
    out.mkdir()

    tf_config = {
        **real_tf,
        "num_layers": 2,
        "num_attention_heads": 2,
        "attention_head_dim": 128,
        "adaln_lora_dim": 32,
        "text_embed_dim": TE_DIM,
        **spec["tf_overrides"],
    }
    torch.manual_seed(0)
    CosmosTransformer3DModel(**tf_config).to(torch.bfloat16).save_pretrained(out / "transformer")
    torch.manual_seed(0)
    AutoencoderKLWan(**real_vae).to(torch.bfloat16).save_pretrained(out / "vae")
    torch.manual_seed(0)
    T5EncoderModel(te_config).to(torch.bfloat16).save_pretrained(out / "text_encoder")

    shutil.copytree(T2I_SNAP / "tokenizer", out / "tokenizer", ignore=shutil.ignore_patterns("tokenizer.pth"))
    (out / "scheduler").mkdir()
    shutil.copy(T2I_SNAP / "scheduler" / "scheduler_config.json", out / "scheduler")

    index = {**model_index, "_class_name": spec["class_name"], "safety_checker": [None, None]}
    (out / "model_index.json").write_text(json.dumps(index, indent=2))

    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"saved {out} ({size / 1e6:.1f} MB)")
