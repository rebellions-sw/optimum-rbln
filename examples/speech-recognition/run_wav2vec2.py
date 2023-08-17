import os

import fire
import torch
from datasets import load_dataset
from transformers import Wav2Vec2Processor

from optimum.rbln import RBLNWav2Vec2ForCTC


def main(
    model_id: str = "facebook/wav2vec2-base-960h",
    batch_size: int = 1,
    from_transformers: bool = False,
):
    processor = Wav2Vec2Processor.from_pretrained(model_id)

    ds = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation")

    input_values = []
    for i in range(batch_size):
        input_value = processor(
            ds[i]["audio"]["array"], return_tensors="pt", padding="max_length", max_length=160005, truncation=True
        ).input_values
        input_values.append(input_value)
    input_values = torch.cat(input_values, dim=0)

    if from_transformers:
        model = RBLNWav2Vec2ForCTC.from_pretrained(
            model_id=model_id,
            export=True,
            rbln_batch_size=batch_size,
            rbln_max_seq_len=160005,
        )
        model.save_pretrained(os.path.basename(model_id))
    else:
        model = RBLNWav2Vec2ForCTC.from_pretrained(model_id=os.path.basename(model_id), export=False)

    output = model(input_values)
    predicted_ids = torch.argmax(output.logits, dim=-1)
    transcriptions = processor.batch_decode(predicted_ids)
    for i, transcription in enumerate(transcriptions):
        print(f"transcription_{i} : {transcription}")


if __name__ == "__main__":
    fire.Fire(main)
