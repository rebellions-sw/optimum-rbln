import torch
from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
from peft.tuners.lora.layer import Linear as LoraLinear
from PIL import Image

from optimum.rbln import RBLNColQwen2_5ForConditionalGeneration


model = ColQwen2_5.from_pretrained(
    "Metric-AI/colqwen2.5-3b-multilingual",
    torch_dtype=torch.float32,
    device_map="cpu",
).eval()
processor = ColQwen2_5_Processor.from_pretrained("Metric-AI/colqwen2.5-3b-multilingual")


for m in model.modules():
    if isinstance(m, LoraLinear):
        m.merge(safe_merge=False)

model = RBLNColQwen2_5ForConditionalGeneration.from_model(
    model,
    export=True,
    rbln_config={
        "visual": {
            "max_seq_lens": 768,
            "device": 0,
        },
        "tensor_parallel_size": 4,
        "max_seq_len": 114_688,
        "kvcache_partition_len": 16_384,
    },
)
model.save_pretrained("colqwen2.5-3b-multilingual")

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
