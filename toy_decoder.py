import pickle

import fire
import scipy
import torch
from transformers import GroundingDinoForObjectDetection

from optimum.rbln import RBLNGroundingDinoDecoder


def main(compile: bool = False, layers: int = 6):
    model_id = "IDEA-Research/grounding-dino-tiny"
    model = GroundingDinoForObjectDetection.from_pretrained(model_id, decoder_layers=layers)

    if compile:
        rbln_decoder = RBLNGroundingDinoDecoder.from_model(model.model.decoder)
        rbln_decoder.save_pretrained("decoder")
    else:
        rbln_decoder = RBLNGroundingDinoDecoder.from_pretrained("decoder", export=False)

    decoder_kwargs = pickle.load(open("/mnt/shared_data/groups/sw_dev/thkim/grounding_dino/decoder_kwargs.pkl", "rb"))

    with torch.inference_mode():
        rbln_output = rbln_decoder(
            **decoder_kwargs,
        )

        golden_model = model.model.decoder
        decoder_kwargs["output_attentions"] = True
        golden_output = golden_model(
            **decoder_kwargs,
        )

    for i, key_name in enumerate(golden_output.keys()):
        try:
            print(f"Result {key_name}")
            max_l1_idff = (rbln_output[i] - golden_output[i]).abs().max()
            pearsonr = scipy.stats.pearsonr(
                rbln_output[i].flatten().cpu().numpy(),
                golden_output[i].flatten().cpu().numpy(),
            )
            print(f"\tMax L1 Diff: {max_l1_idff}")
            print(f"\tPearson Correlation: {pearsonr.correlation}, p-value: {pearsonr.pvalue}")
        except Exception as e:
            print(f"Error processing {key_name}: {e}")
            for r, g in zip(rbln_output[i], golden_output[i]):
                pearsonr = scipy.stats.pearsonr(
                    torch.clip(r, min=-1e5, max=1e5).flatten().numpy(),
                    torch.clip(g[0], min=-1e5, max=1e5).flatten().numpy(),
                )
                print(f"\tPearson Correlation: {pearsonr.correlation}")


if __name__ == "__main__":
    fire.Fire(main)
