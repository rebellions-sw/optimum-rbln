import os

import fire
import torch
from huggingface_hub import hf_hub_download

from optimum.rbln import RBLNTimeSeriesTransformerForPrediction


def main(
    model_id: str = "huggingface/time-series-transformer-tourism-monthly",
    batch_size: int = 1,
    num_parallel_samples: int = 100,
    from_transformers: bool = False,
):
    if from_transformers:
        model = RBLNTimeSeriesTransformerForPrediction.from_pretrained(
            model_id, export=True, rbln_batch_size=batch_size, num_parallel_samples=num_parallel_samples
        )
        model.save_pretrained(os.path.basename(model_id))
    else:
        model = RBLNTimeSeriesTransformerForPrediction.from_pretrained(
            os.path.basename(model_id),
            export=False,
        )

    dataset = hf_hub_download(
        repo_id="hf-internal-testing/tourism-monthly-batch", filename="val-batch.pt", repo_type="dataset"
    )
    data = torch.load(dataset, weights_only=True)

    batched_data = {}
    for k, v in data.items():
        batched_data[k] = v[:batch_size]

    rbln_outputs = model.generate(**batched_data)
    mean_prediction = rbln_outputs.sequences.mean(dim=1)

    print(mean_prediction)


if __name__ == "__main__":
    fire.Fire(main)
