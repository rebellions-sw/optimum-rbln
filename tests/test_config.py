import os
import shutil

import pytest

from optimum.rbln import RBLNResNetForImageClassification, RBLNResNetForImageClassificationConfig
from optimum.rbln.configuration_utils import RBLNCompileConfig


@pytest.fixture
def model_id():
    return "hf-internal-testing/tiny-random-ResNetForImageClassification"


def test_explicit_config_parameters(model_id):
    """Test loading model with explicit configuration parameters."""
    model = RBLNResNetForImageClassification.from_pretrained(
        model_id, export=True, rbln_image_size=224, rbln_batch_size=2, rbln_create_runtimes=False
    )
    assert model is not None
    assert hasattr(model, "rbln_config")
    # Config parameters should be applied correctly


def test_config_dict(model_id):
    """Test loading model with configuration passed as a dictionary."""
    rbln_config = {"create_runtimes": False, "optimize_host_memory": True, "image_size": 64}

    model = RBLNResNetForImageClassification.from_pretrained(model_id, export=True, rbln_config=rbln_config)
    assert model is not None
    assert hasattr(model, "rbln_config")
    assert model.rbln_config.image_size == 64
    # Config dict should be properly applied


def test_config_object(model_id):
    """Test loading model with a pre-configured RBLNResNetForImageClassificationConfig object."""
    config = RBLNResNetForImageClassificationConfig()
    config.create_runtimes = False
    config.image_size = 224

    # Properly set required compile configuration
    compile_cfg = RBLNCompileConfig(input_info=[("pixel_values", (1, 3, 224, 224), "float32")])
    config.set_compile_cfgs([compile_cfg])

    model = RBLNResNetForImageClassification.from_pretrained(model_id, export=True, rbln_config=config)
    assert model is not None
    assert hasattr(model, "rbln_config")
    # Pre-configured object should be properly applied


def test_mixed_config_approach(model_id):
    """Test loading model with both config object and additional parameters."""
    config = RBLNResNetForImageClassificationConfig()
    config.create_runtimes = False

    # Properly set required compile configuration
    compile_cfg = RBLNCompileConfig(input_info=[("pixel_values", (1, 3, 224, 224), "float32")])
    config.set_compile_cfgs([compile_cfg])

    model = RBLNResNetForImageClassification.from_pretrained(
        model_id,
        export=True,
        rbln_config=config,
        rbln_image_size=128,  # This should override the config object
    )
    assert model is not None
    assert hasattr(model, "rbln_config")
    assert model.rbln_config.image_size == 128
    # Check if override parameters were properly applied


def test_config_persistence_after_reload(model_id, tmp_path):
    """Test that configuration values persist correctly after saving and reloading."""
    save_dir = tmp_path / "saved_model"
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)

    os.makedirs(save_dir, exist_ok=True)

    # Use distinctive values to ensure we can detect them
    original_model = RBLNResNetForImageClassification.from_pretrained(
        model_id, export=True, rbln_image_size=112, rbln_batch_size=3, rbln_create_runtimes=False
    )
    original_model.save_pretrained(save_dir)

    # Reload and check
    reloaded_model = RBLNResNetForImageClassification.from_pretrained(save_dir, export=False)

    # Assert specific expected values
    assert reloaded_model.rbln_config.image_size == 112, "image_size configuration was not preserved"
    assert reloaded_model.rbln_config.batch_size == 3, "batch_size configuration was not preserved"
    assert reloaded_model.rbln_config.create_runtimes is True, "create_runtimes configuration should not be preserved"

    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)


def test_config_priority(model_id):
    """Test the priority of different configuration mechanisms."""
    # Create a base config
    config = RBLNResNetForImageClassificationConfig(image_size=224)
    config.create_runtimes = False

    # This explicit parameter should override the config object setting
    model = RBLNResNetForImageClassification.from_pretrained(
        model_id,
        export=True,
        rbln_config=config,
        rbln_image_size=128,  # Should override config.image_size
    )

    assert model.rbln_config.image_size == 128, "Explicit parameter should override config object"
    assert model.rbln_config.create_runtimes is False, "Other config values should be preserved"


@pytest.mark.parametrize(
    "invalid_param",
    [
        {"rbln_nonexistent_param": "value"},
        {"rbln_image_size": "not_an_integer"},  # Type error
        {"rbln_batch_size": -1},  # Negative value
    ],
)
def test_invalid_config_parameters(model_id, invalid_param):
    """Test robust handling of various invalid configuration parameters."""
    with pytest.raises((ValueError, TypeError)):
        _ = RBLNResNetForImageClassification.from_pretrained(model_id, export=True, **invalid_param)


if __name__ == "__main__":
    pytest.main()
