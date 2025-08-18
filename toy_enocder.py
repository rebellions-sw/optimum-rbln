from transformers import GroundingDinoForObjectDetection

from optimum.rbln import RBLNGroundingDinoEncoder
from optimum.rbln.transformers.models.grounding_dino.grounding_dino_architecture import GroundingDinoEncoder


model_id = "IDEA-Research/grounding-dino-tiny"
model = GroundingDinoForObjectDetection.from_pretrained(model_id)


rbln_encoder = RBLNGroundingDinoEncoder.from_model(model.model.encoder)
rbln_encoder.save_pretrained("encoder")

# rbln_encoder = RBLNGroundingDinoEncoder.from_pretrained("encoder", export=False)

import torch


vision_features = torch.randn(1, 37150, 256)
vision_attention_mask = torch.zeros(1, 37150, dtype=torch.float)
vision_position_embedding = torch.randn(1, 37150, 256)
text_features = torch.randn(1, 256, 256)
text_attention_mask = torch.zeros(1, 256, dtype=torch.float)
text_self_attention_masks = torch.zeros(1, 256, 256, dtype=torch.float)
text_position_ids = torch.arange(256, dtype=torch.int32).unsqueeze(0)
reference_points = torch.randn(1, 37150, 4, 2)

golden_model = GroundingDinoEncoder(model.model.encoder)
golden_output = golden_model(
    vision_features,
    vision_attention_mask,
    vision_position_embedding,
    text_features,
    text_attention_mask,
    text_self_attention_masks,
    text_position_ids,
    reference_points,
)

rbln_output = rbln_encoder(
    vision_features,
    vision_attention_mask,
    vision_position_embedding,
    text_features,
    text_attention_mask,
    text_self_attention_masks,
    text_position_ids,
    reference_points,
)
l1_diff = torch.nn.functional.l1_loss(rbln_output.vision_features, golden_output.vision_features)
pearson_corr = torch.corrcoef(rbln_output.vision_features, golden_output.vision_features)
