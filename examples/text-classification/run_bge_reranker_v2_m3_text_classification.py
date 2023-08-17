import os

import fire
from transformers import AutoTokenizer

from optimum.rbln import RBLNXLMRobertaForSequenceClassification


def main(
    model_id: str = "BAAI/bge-reranker-v2-m3",
    from_transformers: bool = False,
    max_seq_len: int = 8192,
    batch_size: int = 1,
):
    if from_transformers:
        model = RBLNXLMRobertaForSequenceClassification.from_pretrained(
            model_id=model_id,
            export=True,
            rbln_max_seq_len=max_seq_len,
            rbln_batch_size=batch_size,
        )
        model.save_pretrained(os.path.basename(model_id))
    else:
        model = RBLNXLMRobertaForSequenceClassification.from_pretrained(
            model_id=os.path.basename(model_id),
            export=False,
        )

    pairs = [
        [
            "what is panda?",
            "The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.",
        ]
    ] * batch_size
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    inputs = tokenizer(pairs, padding="max_length", return_tensors="pt", max_length=max_seq_len)
    score = model(**inputs).view(-1).float()
    print(score)


if __name__ == "__main__":
    fire.Fire(main)
