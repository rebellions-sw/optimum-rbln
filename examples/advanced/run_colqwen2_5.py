import os

import fire
import torch
from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
from modeling_colqwen2_5 import RBLNColQwen2_5ForRetrieval
from configuration_colqwen2_5 import RBLNColQwen2_5ForRetrievalConfig
from peft.tuners.lora.layer import Linear as LoraLinear
from PIL import Image
from scipy.stats import pearsonr

from optimum.rbln import RBLNAutoConfig, RBLNAutoModel


def main(
    model_id: str = "Metric-AI/colqwen2.5-3b-multilingual",
    compile: bool = False,
    diff: bool = False,
):
    # model_id = "/mnt/shared_data/groups/sw_dev/.cache/huggingface/hub/models--Metric-AI--colqwen2.5-3b-multilingual/snapshots/474bd38be82608174dcdfafdd4e801aa7c51b45d/"
    RBLNAutoModel.register(RBLNColQwen2_5ForRetrieval, exist_ok=True)
    RBLNAutoConfig.register(RBLNColQwen2_5ForRetrievalConfig, exist_ok=True)

    if compile or diff:
        hf_model = ColQwen2_5.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            device_map="cpu",
        ).eval()
        processor = ColQwen2_5_Processor.from_pretrained(model_id)

        for m in hf_model.modules():
            if isinstance(m, LoraLinear):
                m.merge(safe_merge=False)

    if compile:
        model = RBLNColQwen2_5ForRetrieval.from_model(
            hf_model,
            export=True,
            rbln_config={
                "visual": {
                    "max_seq_lens": 768,
                    "device": 0,
                },
                "tensor_parallel_size": 4,
                "max_seq_len": 32768,
                # "kvcache_partition_len": 16_384,
                "prefill_chunk_size": 512,
            },
        )
        model.save_pretrained(os.path.basename("colqwen2.5-3b-multilingual"))
    else:
        model = RBLNColQwen2_5ForRetrieval.from_pretrained(
            os.path.basename("Metric-AI/colqwen2.5-3b-multilingual"), export=False
        )

    # Your inputs
    images = [
        Image.new("RGB", (32, 32), color="white"),
        Image.new("RGB", (32, 32), color="black"),
    ]
    queries = [
        "What is the amount of bananas farmed in Salvador?",
    ]

    # Process the inputs
    batch_images = processor.process_images(images).to(model.device)
    batch_queries = processor.process_queries(queries).to(model.device)

    # Forward pass
    with torch.no_grad():
        rbln_image_embeddings = model(**batch_images)
        rbln_query_embeddings = model(**batch_queries)

    scores = processor.score_multi_vector(rbln_query_embeddings, rbln_image_embeddings)
    print(scores)

    if diff:
        # hf_image_embeddings = hf_model(**batch_images)
        hf_query_embeddings = hf_model(**batch_queries)
        # hf_scores = processor.score_multi_vector(hf_query_embeddings, hf_image_embeddings)
        # print(hf_scores)

        hf_query = hf_query_embeddings.detach().cpu().numpy()
        rbln_query = rbln_query_embeddings.detach().cpu().numpy()
        i = 0
        print(pearsonr(hf_query[:, i : i + 128].flatten(), rbln_query[:, i : i + 128].flatten()))
        for i in range(0, hf_query.shape[1]):
            p_value = pearsonr(hf_query[:, i].flatten(), rbln_query[:, i].flatten()).statistic
            if p_value < 0.90:
                print(f"{i}: {p_value} , token : {processor.decode(batch_queries.input_ids[:, i])}")

        breakpoint()


if __name__ == "__main__":
    fire.Fire(main)
