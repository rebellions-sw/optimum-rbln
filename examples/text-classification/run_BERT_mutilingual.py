import os

import fire
import torch
import numpy as np
from transformers import BertTokenizer
from optimum.rbln import RBLNBertForMaskedLM

# Function to score and predict the masked words in a sentence
# https://github.com/huggingface/transformers/blob/6b550462139655d488d4c663086a63e98713c6b9/src/transformers/pipelines/fill_mask.py#L131
def postprocess(tokenizer, model_outputs, input_ids, top_k=5, target_ids=None):
    
    # Cap top_k if there are targets
    if target_ids is not None and target_ids.shape[0] < top_k:
        top_k = target_ids.shape[0]
    outputs = model_outputs

    masked_index = torch.nonzero(input_ids == tokenizer.mask_token_id, as_tuple=False).squeeze(-1)
    # Fill mask pipeline supports only one ${mask_token} per sample

    logits = outputs[0, masked_index, :]
    probs = logits.softmax(dim=-1)
    if target_ids is not None:
        probs = probs[..., target_ids]

    values, predictions = probs.topk(top_k)

    result = []
    single_mask = values.shape[0] == 1
    for i, (_values, _predictions) in enumerate(zip(values.tolist(), predictions.tolist())):
        row = []
        for v, p in zip(_values, _predictions):
            # Copy is important since we're going to modify this array in place
            tokens = input_ids.numpy().copy()
            if target_ids is not None:
                p = target_ids[p].tolist()

            tokens[masked_index[i]] = p
            # Filter padding out:
            tokens = tokens[np.where(tokens != tokenizer.pad_token_id)]
            # Originally we skip special tokens to give readable output.
            # For multi masks though, the other [MASK] would be removed otherwise
            # making the output look odd, so we add them back
            sequence = tokenizer.decode(tokens, skip_special_tokens=single_mask)
            proposition = {"score": v, "token": p, "token_str": tokenizer.decode([p]), "sequence": sequence}
            row.append(proposition)
        result.append(row)
    if single_mask:
        return result[0]
    return result


def main(
    model_id: str = "google-bert/bert-base-multilingual-cased",
    from_transformers: bool = True,
    max_seq_len: int = 512,
    batch_size: int = 1,
):
    if from_transformers:        
        model = RBLNBertForMaskedLM.from_pretrained(
            model_id=model_id,
            export=True,
            rbln_max_seq_len=max_seq_len,
            rbln_batch_size=batch_size,
        )
        model.save_pretrained(os.path.basename(model_id))
    else:
        model = RBLNBertForMaskedLM.from_pretrained(
            model_id=os.path.basename(model_id),
            export=False,
        )
    tokenizer = BertTokenizer.from_pretrained('google-bert/bert-base-multilingual-cased')
    text = ["Hello I'm a [MASK] model."]
    inputs = tokenizer(text, return_tensors='pt', padding="max_length", max_length=max_seq_len)
    output = model(inputs.input_ids, inputs.attention_mask, inputs.token_type_ids)
    result = postprocess(tokenizer, output, inputs.input_ids[0])
    print(result)

if __name__ == "__main__":
    fire.Fire(main)
