import torch
from PIL import Image

from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
from optimum.rbln import RBLNColQwen2_5ForConditionalGeneration

model = ColQwen2_5.from_pretrained(
        "Metric-AI/colqwen2.5-3b-multilingual",
        torch_dtype=torch.float32,
        device_map="cpu",  # or "mps" if on Apple Silicon
    ).eval()
processor = ColQwen2_5_Processor.from_pretrained("Metric-AI/colqwen2.5-3b-multilingual")

from peft.tuners.lora.layer import Linear as LoraLinear
for m in model.modules():
    if isinstance(m, LoraLinear):
        m.merge(safe_merge=True)

model = RBLNColQwen2_5ForConditionalGeneration.from_model(
    model,
    export=True,
    rbln_config={
        # The `device` parameter specifies the device allocation for each submodule during runtime.
        # As Qwen2.5-VL consists of multiple submodules, loading them all onto a single device may exceed its memory capacity, especially as the batch size increases.
        # By distributing submodules across devices, memory usage can be optimized for efficient runtime performance.
        "visual": {
            # Max sequence length for Vision Transformer (ViT), representing the number of patches in an image.
            # Example: For a 224x196 pixel image with patch size 14 and window size 112,
            # the width is padded to 224, resulting in a 224x224 image.
            # This produces 256 patches [(224/14) * (224/14)]. Thus, max_seq_len must be at least 256.
            # For window-based attention, max_seq_len must be a multiple of (window_size / patch_size)^2, e.g., (112/14)^2 = 64.
            # Hence, 256 (64 * 4) is valid. RBLN optimization processes inference per image or video frame, so set max_seq_len to
            # match the maximum expected resolution to optimize computation.
            # "max_seq_lens": 6400,
            "max_seq_lens": 768,
            # The `device` parameter specifies which device should be used for each submodule during runtime.
            "device": 0,
        },
        "tensor_parallel_size": 8,
        "kvcache_partition_len": 16_384,
        # Max position embedding for the language model, must be a multiple of kvcache_partition_len.
        "max_seq_len": 114_688,
        "device": [0, 1, 2, 3, 4, 5, 6, 7],
    },
)


# Your inputs
images = [
    Image.new("RGB", (32, 32), color="white"),
    Image.new("RGB", (32, 32), color="black"),
]
queries = [
    "Is attention really all you need?",
    "What is the amount of bananas farmed in Salvador?",
]

# Process the inputs
batch_images = processor.process_images(images).to(model.device)
batch_queries = processor.process_queries(queries).to(model.device)

# Forward pass
with torch.no_grad():
    image_embeddings = model(**batch_images)
    query_embeddings = model(**batch_queries)

scores = processor.score_multi_vector(query_embeddings, image_embeddings)
print(scores)
    
with torch.no_grad():
    image_embeddings = model(**batch_images)
    query_embeddings = model(**batch_queries)

scores = processor.score_multi_vector(query_embeddings, image_embeddings)
print(scores)

import pdb; pdb.set_trace()