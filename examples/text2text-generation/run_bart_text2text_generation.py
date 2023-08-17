import os

import fire
from transformers import BartTokenizer

from optimum.rbln import RBLNBartForConditionalGeneration


sentences = ["UN Chief Says There Is No <mask> in Syria"]


def main(
    model_id: str = "lucadiliello/bart-small",
    batch_size: int = 1,
    from_transformers: bool = False,
):
    if from_transformers:
        # Compile the RBLN-optimized Bart model (if export=True)
        model = RBLNBartForConditionalGeneration.from_pretrained(
            model_id=model_id,
            export=True,
            # The following arguments are specific to RBLN compilation
            rbln_batch_size=batch_size,
        )
        model.save_pretrained(os.path.basename(model_id))
    else:
        # Load compiled model
        model = RBLNBartForConditionalGeneration.from_pretrained(
            model_id=os.path.basename(model_id),
            export=False,
        )

    # Prepare inputs
    target_sentences = sentences * batch_size
    tokenizer = BartTokenizer.from_pretrained(model_id)
    inputs = tokenizer(target_sentences, return_tensors="pt", padding=True)

    # Generate
    output_sequence = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=50,
        num_beams=1,
    )

    # Decode and print the model's responses
    for i, sentence in enumerate(target_sentences):
        print("\033[94m" + sentence + " : \033[0m\n" + tokenizer.decode(output_sequence.numpy().tolist()[i]))


if __name__ == "__main__":
    fire.Fire(main)
