"""
# Advanced Tutorial: Creating Custom RBLN Classes

This tutorial demonstrates how to create custom RBLN model classes for use with the optimum-rbln framework.

## Requirements for Custom Class Creation:

1. You must create two classes:
   - A custom model class that inherits from RBLNModel
   - A custom config class that inherits from RBLNModelConfig

2. The naming convention is critical:
   - Model class: RBLN<OriginalModelName> (e.g., RBLNResNetModel)
   - Config class: RBLN<OriginalModelName>Config (e.g., RBLNResNetModelConfig)

3. Your custom class MUST implement the _update_rbln_config method
   - This method configures input shapes and compilation settings

4. You must register both classes with the auto registration system
   - This enables the framework to discover and use your custom classes

Important: Custom RBLN classes can only be created for models that exist in
the transformers or diffusers libraries. You cannot create custom RBLN classes
for completely custom architectures without a corresponding HuggingFace implementation.

This example demonstrates creating a custom RBLN class for the ResNet model from transformers.
"""

from typing import TYPE_CHECKING, Optional, Tuple, Union

import torch
from transformers import ResNetModel  # noqa: F401
from transformers.models.resnet.modeling_resnet import BaseModelOutputWithPoolingAndNoAttention

from optimum.rbln import RBLNAutoConfig, RBLNAutoModel, RBLNCompileConfig, RBLNModel, RBLNModelConfig


if TYPE_CHECKING:
    from transformers import AutoFeatureExtractor, AutoProcessor, AutoTokenizer, PretrainedConfig, PreTrainedModel


# STEP 1: Create a custom model class that extends RBLNModel
# ---------------------------------------------------------
# Custom RBLNModel class must follow the naming pattern: RBLN<OriginalModelName>
# In this case, the original model is ResNetModel from transformers
class RBLNResNetModel(RBLNModel):
    """
    Custom RBLN implementation for ResNet model.

    The naming convention is important: RBLN<OriginalModelName> where OriginalModelName
    is the name of the model from transformers or diffusers library.

    This class extends RBLNModel, which is the base class for all RBLN models.
    """

    @classmethod
    def _update_rbln_config(
        cls,
        preprocessors: Optional[Union["AutoFeatureExtractor", "AutoProcessor", "AutoTokenizer"]] = None,
        model: Optional["PreTrainedModel"] = None,
        model_config: Optional["PretrainedConfig"] = None,
        rbln_config: Optional["RBLNResNetModelConfig"] = None,
    ) -> "RBLNResNetModelConfig":
        """
        Updates the RBLN configuration with model-specific settings.

        This method is required and is called during model initialization to properly configure
        the RBLN model for compilation.

        Args:
            preprocessors: Preprocessors from transformers (feature extractor, processor, tokenizer)
            model: The original HuggingFace model
            model_config: Configuration of the original model
            rbln_config: RBLN-specific configuration

        Returns:
            Updated RBLN configuration
        """
        # Set image_size if not provided, obtaining it from the model_config
        if rbln_config.image_size is None:
            if rbln_config.image_size is None:
                rbln_config.image_size = model_config.image_size

            if rbln_config.image_size is None:
                raise ValueError("`image_size` should be specified!")

        # Define the input information for the RBLN compiler
        # For ResNet, the input is 'pixel_values' with shape [batch_size, 3, height, width]
        # The '3' represents RGB channels
        input_info = [
            (
                "pixel_values",
                [rbln_config.batch_size, 3, rbln_config.image_size[0], rbln_config.image_size[1]],
                "float32",
            )
        ]

        # Set the compilation configuration with the defined input info
        rbln_config.set_compile_cfgs([RBLNCompileConfig(input_info=input_info)])
        return rbln_config

    def forward(self, pixel_values, return_dict: Optional[bool] = None, **kwargs):
        """
        Args:
            pixel_values: Input image tensor of shape [batch_size, 3, height, width]
            return_dict: Whether to return output as a dictionary

        Returns:
            Either the raw output tensors or a BaseModelOutputWithPoolingAndNoAttention object
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Execute the compiled RBLN model
        # self.model is a list of rebel.Runtime objects
        # See https://docs.rbln.ai/software/api/python/python_api.html#rebel.rebel_runtime.Runtime for more details
        output = self.model[0](pixel_values)

        # Format the output according to the return_dict flag
        # This ensures compatibility with the original model's output format
        if not return_dict:
            return output
        else:
            return BaseModelOutputWithPoolingAndNoAttention(last_hidden_state=output[0], pooler_output=output[1])


# STEP 2: Create a custom configuration class that extends RBLNModelConfig
# -----------------------------------------------------------------------
# Custom configuration class must follow the pattern: RBLN<OriginalModelName>Config
class RBLNResNetModelConfig(RBLNModelConfig):
    def __init__(self, batch_size: int = None, image_size: Optional[Tuple[int, int]] = None, **kwargs):
        super().__init__(**kwargs)
        self.batch_size = batch_size or 1
        self.image_size = image_size or (224, 224)


# STEP 3: Register both classes with the auto registration system
# --------------------------------------------------------------
# This registration is crucial for the following reasons:
# 1. It enables the RBLN auto-discovery system to find and instantiate our custom classes
# 2. It allows using RBLNAutoModel.from_pretrained() with our custom model type
# 3. It connects our configuration class to the model class in the internal mappings
RBLNAutoModel.register(RBLNResNetModel)
RBLNAutoConfig.register(RBLNResNetModelConfig)


# STEP 4: Usage Example - Creating and using the custom RBLN model
# ----------------------------------------------------------------
# Initialize the model from a pretrained HuggingFace model
# The 'export=True' parameter triggers the model compilation process
# rbln_image_size and rbln_batch_size are passed to the custom configuration
my_model = RBLNResNetModel.from_pretrained(
    "microsoft/resnet-50", export=True, rbln_image_size=(224, 224), rbln_batch_size=1
)

# Save the compiled model for later use
my_model.save_pretrained("my_resnet_model_saved")

# Create a random input tensor for demonstration
# The shape must match what we defined in _update_rbln_config
random_image_input = torch.randn(1, 3, 224, 224)

# Run inference with the compiled model
output = my_model(random_image_input)

# Print each key-value pair in the output dictionary
for key, value in output.items():
    print(key, value)


# STEP 5: Loading a saved RBLN model
# ----------------------------------
# Load the model we just saved - no need to recompile
my_model_reloaded = RBLNResNetModel.from_pretrained("my_resnet_model_saved")
print(my_model_reloaded)
