import rebel
from diffusers.pipelines.cosmos.cosmos_guardrail import RetinaFaceFilter
from diffusers.pipelines.cosmos.cosmos_guardrail import CosmosSafetyChecker
from optimum.rbln.diffusers.pipelines.cosmos.cosmos_guardrail import RBLNCosmosSafetyChecker, RBLNVideoContentSafetyFilter
# from .cosmos_guardrail import RBLNCosmosSafetyChecker
import torch
import numpy as np

if __name__ == "__main__":
    rbln_config = {"batch_size":1}
    video = torch.randn(1, 3, 121, 704, 1280, dtype=torch.float32)
    
    model = CosmosSafetyChecker()
    
    checker = RBLNCosmosSafetyChecker._compile_submodules(
        model = model,
        rbln_config=rbln_config,
        model_save_dir="safety_checker"
    )
    
    from diffusers.video_processor import VideoProcessor
    vid_processor = VideoProcessor(vae_scale_factor=8)
    video = vid_processor.postprocess_video(video, output_type="np")
    video = (video * 255).astype(np.uint8)
    output = checker.check_video_safety(video[0])
    if not output :
        print("video is blocked")
    
    prompt = "A sleek, humanoid robot stands in a vast warehouse filled with neatly stacked cardboard boxes on industrial shelves. The robot's metallic body gleams under the bright, even lighting, highlighting its futuristic design and intricate joints. A glowing blue light emanates from its chest, adding a touch of advanced technology. The background is dominated by rows of boxes, suggesting a highly organized storage system. The floor is lined with wooden pallets, enhancing the industrial setting. The camera remains static, capturing the robot's poised stance amidst the orderly environment, with a shallow depth of field that keeps the focus on the robot while subtly blurring the background for a cinematic effect."
    output = checker.check_text_safety(prompt)
    if not output :
        print("text is blocked")
    