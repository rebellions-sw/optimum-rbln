from transformers import GroundingDinoForObjectDetection

from optimum.rbln import RBLNGroundingDinoDecoder


model_id = "IDEA-Research/grounding-dino-tiny"
model = GroundingDinoForObjectDetection.from_pretrained(model_id)


rbln_model = RBLNGroundingDinoDecoder.from_model(model.model.decoder)
rbln_model.save_pretrained("decoder")
