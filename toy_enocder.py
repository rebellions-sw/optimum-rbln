from transformers import GroundingDinoForObjectDetection
from optimum.rbln import RBLNGroundingDinoEncoder


model_id = "IDEA-Research/grounding-dino-tiny"
model = GroundingDinoForObjectDetection.from_pretrained(model_id)


rbln_model = RBLNGroundingDinoEncoder.from_model(model.model.encoder)
rbln_model.save_pretrained("encoder")