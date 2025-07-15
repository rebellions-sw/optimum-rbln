from transformers import PixtralVisionModel

from transformers import AutoTokenizer, AutoProcessor, AutoModelForImageTextToText, LlavaForConditionalGeneration
from optimum.rbln import RBLNPixtralVisionModel, RBLNLlavaForConditionalGeneration
import fire
import typing
import os
from PIL import Image
import requests

def main(
    model_id: str = "llava-hf/llava-1.5-7b-hf",
    batch_size: int = 1,
    from_transformers: bool = False,
    max_seq_len: typing.Optional[int] = None,
    tensor_parallel_size: typing.Optional[int] = 4,
):    
    base_model = LlavaForConditionalGeneration.from_pretrained(model_id)
    processor = AutoProcessor.from_pretrained(model_id)

    # Define a chat history and use `apply_chat_template` to get correctly formatted prompt
    # Each value in "content" has to be a list of dicts with types ("text", "image") 
    conversation = [
        {

        "role": "user",
        "content": [
            {"type": "text", "text": "What are these?"},
            {"type": "image"},
            {"type": "image"},
            ],
        },
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

    image_file = "http://images.cocodataset.org/val2017/000000039769.jpg"
    raw_image = Image.open(requests.get(image_file, stream=True).raw)
    inputs = processor(images=[raw_image, raw_image], text=prompt, return_tensors='pt')

    if from_transformers:
        model = RBLNLlavaForConditionalGeneration.from_pretrained(
            model_id=model_id,
            export=True,
            rbln_batch_size=batch_size,
            rbln_config={
                "language_model":
                    {
                        "use_inputs_embeds": True,
                    }
                
            }   
        )
        model.save_pretrained(os.path.basename(model_id))
    else:
        model = RBLNLlavaForConditionalGeneration.from_pretrained(model_id=os.path.basename(model_id), export=False)

    
    output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
    print(processor.decode(output[0][2:], skip_special_tokens=True))
    
    golden_output = base_model.generate(**inputs, max_new_tokens=200, do_sample=False)
    print(processor.decode(golden_output[0][2:], skip_special_tokens=True))

if __name__ == "__main__":
    fire.Fire(main)