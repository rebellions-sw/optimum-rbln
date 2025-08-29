import os

import fire
from transformers import pipeline

from optimum.rbln import RBLNBertForQuestionAnswering


def main(
    model_id: str = "deepset/bert-base-cased-squad2",  # or "deepset/roberta-base-squad2"
    from_transformers: bool = False,
    batch_size: int = 1,
):
    if from_transformers:
        model = RBLNBertForQuestionAnswering.from_pretrained(
            model_id=model_id,
            export=True,
            rbln_max_seq_len=512,
            rbln_batch_size=batch_size,
        )
        model.save_pretrained(os.path.basename(model_id))
    else:
        model = RBLNBertForQuestionAnswering.from_pretrained(
            model_id=os.path.basename(model_id),
            export=False,
        )

    pipe = pipeline(
        "question-answering",
        model=model,
        tokenizer=model_id,
        padding="max_length",
        max_seq_len=model.rbln_config.max_seq_len,
    )
    question, text = (
        ["What is Rebellions?"] * batch_size,
        ["Rebellions is the best NPU company."] * batch_size,
    )
    print(pipe(question=question, context=text, batch_size=batch_size))


if __name__ == "__main__":
    fire.Fire(main)
