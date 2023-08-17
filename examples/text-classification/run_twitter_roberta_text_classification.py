import csv
import os
import urllib.request

import fire
import numpy as np
import torch
from transformers import RobertaTokenizerFast

from optimum.rbln import RBLNRobertaForSequenceClassification


NUM_CLASS = 4


# Preprocess text
def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = "@user" if t.startswith("@") and len(t) > 1 else t
        t = "http" if t.startswith("http") else t
        new_text.append(t)
    return " ".join(new_text)


def download_label_mapping(task):
    mapping_link = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/{task}/mapping.txt"
    with urllib.request.urlopen(mapping_link) as f:
        html = f.read().decode("utf-8").split("\n")
        csvreader = csv.reader(html, delimiter="\t")
    return [row[1] for row in csvreader if len(row) > 1]


def predict(text, tokenizer, model, max_seq_len, labels):
    # Encode the text
    text[0] = preprocess(text[0])
    inputs = tokenizer.batch_encode_plus(
        text, max_length=max_seq_len, truncation=True, padding="max_length", return_tensors="pt"
    )

    # Run the model
    output = model(inputs.input_ids, inputs.attention_mask)

    for batch_itr in range(output.shape[0]):
        # Apply softmax to get probabilities
        scores = output[batch_itr].detach()
        scores = torch.nn.functional.softmax(scores, dim=-1).numpy()

        # Get ranking of scores
        ranking = np.argsort(scores)
        ranking = ranking[::-1]
        # Print out the results
        for i in range(scores.shape[0]):
            l = labels[ranking[i]]
            s = scores[ranking[i]]
            print(f"{batch_itr}) {l} {np.round(float(s), NUM_CLASS)}")


def main(
    model_id: str = None,
    from_transformers: bool = True,
    max_seq_len: int = 512,
    batch_size: int = 1,
):
    task = "emotion"
    model_id = f"cardiffnlp/twitter-roberta-base-{task}"

    if from_transformers:
        model = RBLNRobertaForSequenceClassification.from_pretrained(
            model_id=model_id,
            export=True,
            rbln_max_seq_len=max_seq_len,
            rbln_batch_size=batch_size,
        )
        model.save_pretrained(os.path.basename(model_id))
    else:
        model = RBLNRobertaForSequenceClassification.from_pretrained(
            model_id=os.path.basename(model_id),
            export=False,
        )

    prompt = ["Celebrating my promotion 😎"]

    target_sentences = prompt

    tokenizer = RobertaTokenizerFast.from_pretrained(model_id)
    labels = download_label_mapping(task)
    predict(target_sentences, tokenizer, model, max_seq_len, labels)


if __name__ == "__main__":
    fire.Fire(main)
