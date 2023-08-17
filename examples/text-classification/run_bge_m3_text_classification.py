import os

import fire
import torch
from transformers import AutoTokenizer

from optimum.rbln import RBLNXLMRobertaModel


def main(
    model_id: str = "BAAI/bge-m3",
    from_transformers: bool = False,
    max_seq_len: int = 8192,
    batch_size: int = 1,
):
    if from_transformers:
        model = RBLNXLMRobertaModel.from_pretrained(
            model_id=model_id,
            export=True,
            rbln_max_seq_len=max_seq_len,
            rbln_batch_size=batch_size,
        )
        model.save_pretrained(os.path.basename(model_id))
    else:
        model = RBLNXLMRobertaModel.from_pretrained(
            model_id=os.path.basename(model_id),
            export=False,
        )

    input_q = ["what is panda?"] * batch_size
    input_m = [
        "The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China."
    ] * batch_size
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    input_q = tokenizer(input_q, padding="max_length", return_tensors="pt", max_length=max_seq_len)
    input_m = tokenizer(input_m, padding="max_length", return_tensors="pt", max_length=max_seq_len)

    q_output = model(input_q.input_ids, input_q.attention_mask)
    m_output = model(input_m.input_ids, input_m.attention_mask)
    q_output = torch.nn.functional.normalize(q_output[0][:, 0], dim=-1)
    m_output = torch.nn.functional.normalize(m_output[0][:, 0], dim=-1)

    score = q_output @ m_output.T
    print(score)


if __name__ == "__main__":
    fire.Fire(main)
