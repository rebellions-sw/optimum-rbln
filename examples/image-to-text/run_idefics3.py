import os
import typing

import fire
from transformers.image_utils import load_image
from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers import Idefics3VisionTransformer
from optimum.rbln import RBLNIdefics3VisionTransformer, RBLNIdefics3ForConditionalGeneration, RBLNIdefics3Model
# from optimum.rbln import RBLNIdefics3ForConditionalGeneration, RBLNIdefics3VisionTransformer


def main(
    model_id: str = "HuggingFaceM4/Idefics3-8B-Llama3",
    batch_size: int = 1,
    from_transformers: bool = False,
    prompt: typing.Optional[str] = None,
    max_seq_len: typing.Optional[int] = None,
    tensor_parallel_size: typing.Optional[int] = 4,
    num_text_only: typing.Optional[int] = None,
):
    processor = AutoProcessor.from_pretrained(model_id)
    # base_model = AutoModelForVision2Seq.from_pretrained(model_id)
    # # vision_model = Idefics3VisionTransformer._from_config(base_model.config.vision_config)
    # tmp_model = RBLNIdefics3Connector.from_model(
    #     base_model.model
    # )
    # import pdb; pdb.set_trace()
    if from_transformers:
        model = RBLNIdefics3ForConditionalGeneration.from_pretrained(
            model_id,
            export=True,
            rbln_attn_impl="flash_attn",
            rbln_max_seq_len=131072,
            rbln_use_inputs_embeds=True,
            rbln_tensor_parallel_size=4,
        )
        model.save_pretrained(os.path.basename(model_id))
    else:
        model = RBLNIdefics3ForConditionalGeneration.from_pretrained(
            os.path.basename(model_id),
            export=False,
        )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "What do we see in this image?"},
            ]
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "In this image, we can see the city of New York, and more specifically the Statue of Liberty."},
            ]
        },
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "And how about this image?"},
            ]
        },       
    ]
    
    image1 = load_image("https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg")
    image2 = load_image("https://cdn.britannica.com/59/94459-050-DBA42467/Skyline-Chicago.jpg")
    image3 = load_image("https://cdn.britannica.com/68/170868-050-8DDE8263/Golden-Gate-Bridge-San-Francisco.jpg")
    
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[image1, image2], return_tensors="pt")
    inputs = {k: v for k, v in inputs.items()}
    
    
    # for vision
    # vision_out = base_model.model.vision_model(inputs['pixel_values'].squeeze(0))
    # print(vision_out.last_hidden_state)
    # print('=============')
    # rbln_out = model(inputs['pixel_values'].squeeze(0))
    # print(rbln_out.last_hidden_state)

    # # compare
    # import numpy as np
    # from scipy import stats
    # golden = vision_out.last_hidden_state.detach().numpy()
    # rbln = rbln_out.last_hidden_state.detach().numpy()
    # print(f"max l1 diff : {np.abs(golden - rbln).max()}")
    # print(f"pearsonr : {stats.pearsonr(golden.flatten(), rbln.flatten()).statistic}")
    
    generated_ids = model.generate(**inputs, max_new_tokens=500)
    generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)

    print(generated_texts)
    # output = model.generate(**inputs, max_new_tokens=200)

    # for i in range(batch_size):
    #     prompt_len = inputs.input_ids[i].shape[-1]
    #     if i >= num_text_only:
    #         images[i - num_text_only].save(f"batch_{i}.png", "png")
    #     print(f"batch {i} -- Prompt --")
    #     print(processor.decode(output[i][:prompt_len], skip_special_tokens=True))
    #     print("---- Answer ----")
    #     print(processor.decode(output[i][prompt_len:], skip_special_tokens=True))


if __name__ == "__main__":
    fire.Fire(main)
