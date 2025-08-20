import torch
import pickle
import fire
from transformers import GroundingDinoForObjectDetection

from optimum.rbln import RBLNGroundingDinoEncoder
import scipy

def main(compile: bool = False, layers: int = 6):

    model_id = "IDEA-Research/grounding-dino-tiny"
    model = GroundingDinoForObjectDetection.from_pretrained(model_id, encoder_layers=layers)

    if compile:
        rbln_encoder = RBLNGroundingDinoEncoder.from_model(model.model.encoder)
        rbln_encoder.save_pretrained("encoder")
    else:
        rbln_encoder = RBLNGroundingDinoEncoder.from_pretrained("encoder", export=False)

    encoder_kwargs = pickle.load(open("/mnt/shared_data/groups/sw_dev/thkim/grounding_dino/encoder_kwargs.pkl", "rb"))
    
    with torch.inference_mode():
        golden_model = model.model.encoder
        golden_output = golden_model(
            **encoder_kwargs,
        )

        rbln_output = rbln_encoder(
            **encoder_kwargs,
        )

    for i, key_name in enumerate(golden_output.keys()):
        print(f"Result {key_name}")
        max_l1_idff = (rbln_output[i] - golden_output[i]).abs().max()
        pearsonr = scipy.stats.pearsonr(
            rbln_output[i].flatten().cpu().numpy(),
            golden_output[i].flatten().cpu().numpy(),
        )
        print(f"\tMax L1 Diff: {max_l1_idff}")
        print(f"\tPearson Correlation: {pearsonr.correlation}, p-value: {pearsonr.pvalue}")

if __name__ == "__main__":
    fire.Fire(main)