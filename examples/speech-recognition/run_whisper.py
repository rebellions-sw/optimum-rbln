import os

import fire
import torch
from datasets import load_dataset
from transformers import AutoProcessor, pipeline

from optimum.rbln import RBLNWhisperForConditionalGeneration


def prepare_shortform(model_id, batch_size):
    processor = AutoProcessor.from_pretrained(model_id)
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation", trust_remote_code=True)
    input_features_list = []

    for i in range(batch_size):
        input_features = processor(
            ds[i]["audio"]["array"],
            sampling_rate=ds[i]["audio"]["sampling_rate"],
            truncation=False,
            return_tensors="pt",
        ).input_features
        input_features_list.append(input_features)
    input_features = torch.cat(input_features_list, dim=0)
    return processor, input_features


def prepare_longform(model_id, batch_size):
    processor = AutoProcessor.from_pretrained(model_id)
    ds = load_dataset("distil-whisper/librispeech_long", "clean", split="validation")

    input = processor(
        ds[0]["audio"]["array"],
        sampling_rate=ds[0]["audio"]["sampling_rate"],
        truncation=False,
        padding="longest",
        return_attention_mask=True,
        return_tensors="pt",
    )

    input_features = input.input_features.repeat(batch_size, 1, 1)
    attention_mask = input.attention_mask.repeat(batch_size, 1)

    return processor, input_features, attention_mask


def main(
    model_id: str = "openai/whisper-tiny",
    # rbln config
    batch_size: int = 1,
    from_transformers: bool = False,
    return_token_timestamps: bool = False,  # valid_only with shortform?
    # generation config
    long_form: bool = False,
    pipe: bool = False,
):
    # set kwargs
    rbln_kwargs = {}
    gen_kwargs = {"return_timestamps": True}
    if return_token_timestamps:
        gen_kwargs.update({"return_token_timestamps": True})
        rbln_kwargs.update({"rbln_token_timestamps": True})

    # compile or load model
    if from_transformers:
        model = RBLNWhisperForConditionalGeneration.from_pretrained(
            model_id=model_id,
            export=True,
            rbln_batch_size=batch_size,
            **rbln_kwargs,
        )
        model.save_pretrained(os.path.basename(model_id))
    else:
        model = RBLNWhisperForConditionalGeneration.from_pretrained(
            model_id=os.path.basename(model_id),
            export=False,
        )

    # generation strategy
    # 1. short_form
    # 2. long_form
    # 3. pipe
    if not long_form and not pipe:
        processor, input_features = prepare_shortform(model_id, batch_size)
        outputs = model.generate(
            input_features=input_features,
            **gen_kwargs,
        )

        generated_ids = outputs["sequences"] if return_token_timestamps else outputs
        transcriptions = processor.batch_decode(generated_ids, skip_special_tokens=True, decode_with_timestamps=True)

        print("---RBLN Shortform Generate Result ---")
        for i, transcription in enumerate(transcriptions):
            print(f"transcription {i} : {transcription}")
            if return_token_timestamps:
                print(f"token_timestamps {i} : {outputs['token_timestamps'][i]}")

    if long_form:
        processor, input_features, attention_mask = prepare_longform(model_id, batch_size)
        outputs = model.generate(
            input_features=input_features,
            attention_mask=attention_mask,
            **gen_kwargs,
        )

        generated_ids = outputs.get("sequences") if isinstance(outputs, dict) else outputs
        transcriptions = processor.batch_decode(generated_ids, skip_special_tokens=True, decode_with_timestamps=True)

        print("---RBLN Longform Generate Result ---")
        for i, transcription in enumerate(transcriptions):
            print(f"transcription {i} : {transcription}")

    if pipe:
        processor = AutoProcessor.from_pretrained(model_id)
        dataset = load_dataset("distil-whisper/librispeech_long", "clean", split="validation")
        sample = dataset[0]["audio"]
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            chunk_length_s=30,
            return_timestamps=True,
            batch_size=batch_size,
        )
        generate_kwargs = {"repetition_penalty": 1.3}

        with torch.no_grad():
            outputs = pipe(sample, generate_kwargs=generate_kwargs)
        print("---RBLN Pipeline Result ---")
        print("--Text--")
        print(outputs["text"])
        print("--Chunks--")
        print(outputs["chunks"])


if __name__ == "__main__":
    fire.Fire(main)
