from transformers import PixtralVisionModel

from transformers import AutoTokenizer, AutoProcessor, AutoModelForImageTextToText
from optimum.rbln import RBLNPixtralVisionModel, RBLNLlavaForConditionalGeneration
import fire
import typing
import os

def main(
    model_id: str = "mistral-community/pixtral-12b",
    batch_size: int = 1,
    from_transformers: bool = False,
    max_seq_len: typing.Optional[int] = None,
    tensor_parallel_size: typing.Optional[int] = 4,
):    
    base_model = AutoModelForImageTextToText.from_pretrained("mistral-community/pixtral-12b")
    processor = AutoProcessor.from_pretrained("mistral-community/pixtral-12b")

    # IMG_URLS = [
    #     "https://picsum.photos/id/237/400/300", 
    # ]
    # PROMPT = "<s>[INST]Describe the images.\n[IMG]"
    
    IMG_URLS = [
        "https://picsum.photos/id/237/400/300", 
        "https://picsum.photos/id/231/200/300", 
        "https://picsum.photos/id/27/500/500",
        "https://picsum.photos/id/17/150/600",
    ]
    
    PROMPT = "<s>[INST]Describe the images.\n[IMG][IMG][IMG][IMG][/INST]"
    inputs = processor(text=PROMPT, images=IMG_URLS, return_tensors="pt")
    # base_model.generate(**inputs, max_new_tokens=100)
    # import pdb; pdb.set_trace()
    
    # if from_transformers:
    #     model = RBLNLlavaForConditionalGeneration.from_pretrained(
    #         model_id=model_id,
    #         export=True,
    #         rbln_batch_size=batch_size,
    #         rbln_config={
    #             "vision_tower":{
    #                 "max_image_size": (304, 400),
    #             },
    #             "language_model":
    #                 {
    #                     "use_inputs_embeds": True,
    #                     "tensor_parallel_size": 8,
    #                     "max_seq_len": 131072,
    #                     "kvcache_partition_len": 16384,
    #                 }
                
    #         }
    #         # rbln_image_size=(304,400),
    #         # rbln_max_seq_len=max_seq_len,
    #         # rbln_tensor_parallel_size=tensor_parallel_size,
    #         # rbln_use_inputs_embeds=True,
    #     )
    #     model.save_pretrained(os.path.basename(model_id))
    # else:
    #     model = RBLNLlavaForConditionalGeneration.from_pretrained(model_id=os.path.basename(model_id), export=False)

    # inputs = processor(text=PROMPT, images=IMG_URLS, return_tensors="pt")
    # model.generate(**inputs)
    # generate_ids = model.generate(**inputs, max_new_tokens=100)
    golden_ids = base_model.generate(**inputs, max_new_tokens=100)
    import pdb; pdb.set_trace()
    
if __name__ == "__main__":
    fire.Fire(main)