import os
import typing

import fire
import numpy as np
import torch
from scipy.stats import pearsonr
from transformers import ColPaliForRetrieval, ColPaliProcessor

from optimum.rbln import RBLNColPaliForRetrieval


def main(
    model_id: str = "vidore/colpali-v1.3-hf",
    batch_size: int = 1,
    from_transformers: bool = False,
    prompt: typing.Optional[str] = None,
    seq_len: typing.Optional[int] = 8192,
    tensor_parallel_size: typing.Optional[int] = 1,
    diff: bool = False,
    run: bool = False,
):
    if from_transformers:
        model = RBLNColPaliForRetrieval.from_pretrained(
            model_id,
            export=True,
            rbln_config={
                "tensor_parallel_size": tensor_parallel_size,
                "max_seq_len": seq_len,
                "output_hidden_states": False,
            },
        )
        model.save_pretrained(os.path.basename(model_id))
    else:
        # processor = AutoProcessor.from_pretrained(os.path.basename(model_id))
        model = RBLNColPaliForRetrieval.from_pretrained(
            os.path.basename(model_id),
            export=False,
        )

    if diff:
        model_hf = ColPaliForRetrieval.from_pretrained(model_id).eval()

    from PIL import Image

    import requests
    processor = ColPaliProcessor.from_pretrained("vidore/colpali-v1.3-hf")

    url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg?download=true"
    # url2 = "https://cdn-lfs.hf.co/datasets/huggingface/documentation-images/1d0308bee9040b2332cb64f82c4b74c23d45c41e14c68e85b557efca16e7fc4f?response-content-disposition=inline%3B+filename*%3DUTF-8%27%27inpaint.png%3B+filename%3D%22inpaint.png%22%3B&response-content-type=image%2Fpng&Expires=1750129259&Policy=eyJTdGF0ZW1lbnQiOlt7IkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTc1MDEyOTI1OX19LCJSZXNvdXJjZSI6Imh0dHBzOi8vY2RuLWxmcy5oZi5jby9kYXRhc2V0cy9odWdnaW5nZmFjZS9kb2N1bWVudGF0aW9uLWltYWdlcy8xZDAzMDhiZWU5MDQwYjIzMzJjYjY0ZjgyYzRiNzRjMjNkNDVjNDFlMTRjNjhlODViNTU3ZWZjYTE2ZTdmYzRmP3Jlc3BvbnNlLWNvbnRlbnQtZGlzcG9zaXRpb249KiZyZXNwb25zZS1jb250ZW50LXR5cGU9KiJ9XX0_&Signature=nFbx7wB%7E%7EnyME-cC-ONQgNm%7E7t1-jm4sOAtIE4sCduId-5F119UhHP5jYfld0MSewAtai01r0o3vPaqWJupx%7EqRCJ51LeSdOeVSIrq9JDntI-BlzmHojwfWvXoMZorpiKJ0wN4HaRocSJu6mWBb5-uPlhTJRq6VwhZUx0%7Eg0OmY4K6z9MeMskUX6jW7RCdm9u4Z2WDj8i8ZGrZKV4IBkT%7EaOgb2Rx3pom7IcgEpE7LM62iSCmekOgrfcQbml%7EVKgaCUfXUG8awe0Qt60TjGcj-u9VgrkgUDAwmsV7HvEoMZ85MNrXCKL60SFqQKpRhhqceyNNuyV0XYhtq-iLRwl6g__&Key-Pair-Id=K3RPWS32NSSJCE"
    image1 = Image.open(requests.get(url, stream=True).raw)
    # # image2 = Image.open(requests.get(url2, stream=True).raw)

    # Your inputs
    images = [
        image1,
        # Image.new("RGB", (16, 16), color="black"),
    ]
    queries = [
        # "What is the organizational structure for our R&D department?",
        "Can you provide a breakdown of last year’s financial performance?",
    ]

    # Process the inputs
    batch_images = processor(images=images)
    batch_queries = processor(text=queries)

    from time import time

    if run:
        # Forward pass
        start_time = time()
        with torch.no_grad():
            image_embeddings = model(**batch_images)
            query_embeddings = model(**batch_queries)
        end_time = time()
        print(f"RBLN TIME: {end_time - start_time}")

        # Score the queries against the images
        scores = processor.score_retrieval(query_embeddings.embeddings, image_embeddings.embeddings)

        print("RBLN RESULT")
        print(scores)
        rbln_image_embeddings = image_embeddings.embeddings
        rbln_query_embeddings = query_embeddings.embeddings
        print(image_embeddings.embeddings)
        print(query_embeddings.embeddings)

    if diff:
        start_time = time()
        with torch.no_grad():
            image_embeddings = model_hf(**batch_images)
            query_embeddings = model_hf(**batch_queries)
            scores = processor.score_retrieval(query_embeddings.embeddings, image_embeddings.embeddings)
        end_time = time()
        print(f"HF TIME: {end_time - start_time}")
        print("HF RESULT")
        print(scores)
        hf_image_embeddings = image_embeddings.embeddings
        hf_query_embeddings = query_embeddings.embeddings
        print(image_embeddings.embeddings)
        print(query_embeddings.embeddings)

        pearson_r = pearsonr(
            np.array(rbln_image_embeddings).flatten(), np.array(hf_image_embeddings).flatten()
        ).statistic
        print(pearson_r)
        pearson_r = pearsonr(
            np.array(rbln_query_embeddings).flatten(), np.array(hf_query_embeddings).flatten()
        ).statistic
        print(pearson_r)


if __name__ == "__main__":
    fire.Fire(main)