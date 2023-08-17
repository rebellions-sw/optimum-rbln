import os

import fire
import torch
from datasets import load_dataset
from transformers import AutoFeatureExtractor

from optimum.rbln import RBLNASTForAudioClassification


def main(
    model_id: str = "MIT/ast-finetuned-audioset-10-10-0.4593",
    batch_size: int = 1,
    from_transformers: bool = False,
):
    if from_transformers:
        model = RBLNASTForAudioClassification.from_pretrained(
            model_id=model_id,
            export=True,
            rbln_batch_size=batch_size,
        )
        model.save_pretrained(os.path.basename(model_id))
    else:
        model = RBLNASTForAudioClassification.from_pretrained(
            model_id=os.path.basename(model_id),
            export=False,
        )

    ds = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation", trust_remote_code=True)
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
    input_values_list = []
    for i in range(batch_size):
        input_values = feature_extractor(ds[i]["audio"]["array"], return_tensors="pt").input_values
        input_values_list.append(input_values)
    input_values = torch.cat(input_values_list, dim=0)

    rbln_logits = model(input_values)
    rbln_labels = []
    for i in range(batch_size):
        rbln_logit = rbln_logits[i]
        rbln_class_ids = torch.argmax(rbln_logit, dim=-1).item()
        rbln_label = model.config.id2label[rbln_class_ids]
        rbln_labels.append(rbln_label)
    print(f"labels : {rbln_labels}")


if __name__ == "__main__":
    fire.Fire(main)
