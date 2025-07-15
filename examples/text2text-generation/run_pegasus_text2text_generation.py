import os

import fire
from transformers import PegasusTokenizer

from optimum.rbln import RBLNPegasusForConditionalGeneration


sentences = [
    "PG&E stated it scheduled the blackouts in response to forecasts for high winds "
    "amid dry conditions. The aim is to reduce the risk of wildfires. Nearly 800 thousand customers were "
    "scheduled to be affected by the shutoffs which were expected to last through at least midday tomorrow."
]

def main(
    model_id: str = "google/pegasus-xsum",
    batch_size: int = 1,
    from_transformers: bool = False,
):
    if from_transformers:
        # Compile the RBLN-optimized Bart model (if export=True)
        model = RBLNPegasusForConditionalGeneration.from_pretrained(
            model_id=model_id,
            export=True,
            # The following arguments are specific to RBLN compilation
            rbln_batch_size=batch_size,
        )
        model.save_pretrained(os.path.basename(model_id))
    else:
        # Load compiled model
        model = RBLNPegasusForConditionalGeneration.from_pretrained(
            model_id=os.path.basename(model_id),
            export=False,
        )

    # Prepare inputs
    target_sentences = sentences * batch_size
    tokenizer = PegasusTokenizer.from_pretrained(model_id)
    inputs = tokenizer(target_sentences, max_length=1024, return_tensors="pt")
    
    # Generate
    output_sequence = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=50,
        num_beams=1,
        do_sample=False,
    )

    # Decode and print the model's responses
    for i, sentence in enumerate(target_sentences):
        print("\033[94m" + sentence + " : \033[0m\n" + tokenizer.decode(output_sequence.numpy().tolist()[i], 
                                                                        skip_special_tokens=True, 
                                                                        clean_up_tokenization_spaces=False))
    
if __name__ == "__main__":
    fire.Fire(main)
