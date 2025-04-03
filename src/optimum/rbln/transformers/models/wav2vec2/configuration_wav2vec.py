from ...configuration_generic import RBLNModelForMaskedLMConfig


class RBLNWav2Vec2ForCTCConfig(RBLNModelForMaskedLMConfig):
    rbln_model_input_names = ["input_values"]
