import unittest

import torch
from diffusers import ControlNetModel

from optimum.rbln import (
    RBLNKandinskyV22InpaintCombinedPipeline,
    RBLNStableDiffusion3Img2ImgPipeline,
    RBLNStableDiffusion3Pipeline,
    RBLNStableDiffusionControlNetPipeline,
    RBLNStableDiffusionImg2ImgPipeline,
    RBLNStableDiffusionPipeline,
    RBLNStableDiffusionXLControlNetPipeline,
    RBLNStableDiffusionXLPipeline,
    RBLNStableVideoDiffusionPipeline,
)

from .test_base import BaseTest


class TestSDModel(BaseTest.TestModel):
    RBLN_CLASS = RBLNStableDiffusionPipeline
    HF_MODEL_ID = "hf-internal-testing/tiny-sd-pipe"
    GENERATION_KWARGS = {
        "prompt": "an illustration of a cute white cat riding a black horse on mars",
        "num_inference_steps": 3,
        "generator": torch.manual_seed(42),
    }
    # Fix incorrect tiny-sd-pipe's vae config.json sample_size
    RBLN_CLASS_KWARGS = {
        "rbln_config": {
            "vae": {
                "sample_size": (64, 64),
            },
            "unet": {
                "batch_size": 2,
            },
        }
    }


class TestSDModelBatch(BaseTest.TestModel):
    RBLN_CLASS = RBLNStableDiffusionPipeline
    HF_MODEL_ID = "hf-internal-testing/tiny-sd-pipe"
    GENERATION_KWARGS = {
        "prompt": ["an illustration of a cute white cat riding a black horse on mars"] * 2,
        "num_inference_steps": 3,
        "generator": torch.manual_seed(42),
        "guidance_scale": 0.0,
    }
    # Fix incorrect tiny-sd-pipe's vae config.json sample_size
    RBLN_CLASS_KWARGS = {
        "rbln_batch_size": 2,
        "rbln_config": {
            "vae": {
                "sample_size": (64, 64),
            },
        },
    }


class TestSDXLModel(BaseTest.TestModel):
    RBLN_CLASS = RBLNStableDiffusionXLPipeline
    HF_MODEL_ID = "hf-internal-testing/tiny-sdxl-pipe"
    GENERATION_KWARGS = {
        "prompt": "an illustration of a cute white cat riding a black horse on mars",
        "num_inference_steps": 3,
        "generator": torch.manual_seed(42),
    }
    # Fix incorrect tiny-sd-pipe-xl's vae config.json sample_size
    RBLN_CLASS_KWARGS = {
        "rbln_config": {
            "vae": {
                "sample_size": (64, 64),
            }
        }
    }


class TestSDImg2ImgModel(BaseTest.TestModel):
    RBLN_CLASS = RBLNStableDiffusionImg2ImgPipeline
    HF_MODEL_ID = "hf-internal-testing/tiny-sd-pipe"
    GENERATION_KWARGS = {
        "prompt": "an illustration of a cute white cat riding a black horse on mars",
        "num_inference_steps": 3,
        "strength": 0.75,
        "generator": torch.manual_seed(42),
        "image": torch.randn(1, 3, 64, 64, generator=torch.manual_seed(42)).clamp(0, 1),
    }
    # Fix incorrect tiny-sd-pipe's vae config.json sample_size
    RBLN_CLASS_KWARGS = {
        "rbln_config": {
            "vae": {
                "sample_size": (64, 64),
            }
        },
        "rbln_img_width": 64,
        "rbln_img_height": 64,
    }


class TestSDControlNetModel(BaseTest.TestModel):
    RBLN_CLASS = RBLNStableDiffusionControlNetPipeline
    HF_MODEL_ID = "hf-internal-testing/tiny-stable-diffusion-torch"
    CONTROLNET_ID = "hf-internal-testing/tiny-controlnet"

    GENERATION_KWARGS = {
        "prompt": "the mona lisa",
        "image": torch.randn(1, 3, 64, 64, generator=torch.manual_seed(42)),
    }
    RBLN_CLASS_KWARGS = {
        "rbln_img_width": 64,
        "rbln_img_height": 64,
    }

    @classmethod
    def setUpClass(cls):
        controlnet = ControlNetModel.from_pretrained(cls.CONTROLNET_ID)
        cls.RBLN_CLASS_KWARGS["controlnet"] = controlnet
        return super().setUpClass()


class TestSDXLControlNetModel(BaseTest.TestModel):
    RBLN_CLASS = RBLNStableDiffusionXLControlNetPipeline
    HF_MODEL_ID = "hf-internal-testing/tiny-sdxl-pipe"
    CONTROLNET_ID = "hf-internal-testing/tiny-controlnet-sdxl"

    GENERATION_KWARGS = {
        "prompt": "aerial view, a futuristic research complex in a bright foggy jungle, hard lighting",
        "image": torch.randn(1, 3, 64, 64, generator=torch.manual_seed(42)),
        "controlnet_conditioning_scale": 0.5,
    }
    RBLN_CLASS_KWARGS = {
        "rbln_img_width": 64,
        "rbln_img_height": 64,
    }

    @classmethod
    def setUpClass(cls):
        controlnet = ControlNetModel.from_pretrained(cls.CONTROLNET_ID)
        cls.RBLN_CLASS_KWARGS["controlnet"] = controlnet
        return super().setUpClass()


class TestSD3Model(BaseTest.TestModel):
    RBLN_CLASS = RBLNStableDiffusion3Pipeline
    HF_MODEL_ID = "hf-internal-testing/tiny-sd3-pipe"
    GENERATION_KWARGS = {
        "prompt": "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
        "num_inference_steps": 3,
        "generator": torch.manual_seed(42),
    }
    RBLN_CLASS_KWARGS = {
        "rbln_config": {
            "text_encoder": {"rbln_device": 0},
            "text_encoder_2": {"rbln_device": 0},
            "text_encoder_3": {"rbln_device": -1},
            "transformer": {"rbln_device": 0},
            "vae": {"rbln_device": 0},
        },
    }


class TestSD3Img2ImgModel(BaseTest.TestModel):
    RBLN_CLASS = RBLNStableDiffusion3Img2ImgPipeline
    HF_MODEL_ID = "hf-internal-testing/tiny-sd3-pipe"
    GENERATION_KWARGS = {
        "prompt": "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
        "generator": torch.manual_seed(42),
        "strength": 0.95,
        "image": torch.randn(1, 3, 64, 64, generator=torch.manual_seed(42)),
    }
    RBLN_CLASS_KWARGS = {
        "rbln_img_width": 64,
        "rbln_img_height": 64,
        "rbln_config": {
            "text_encoder": {"rbln_device": 0},
            "text_encoder_2": {"rbln_device": 0},
            "text_encoder_3": {"rbln_device": -1},
            "transformer": {"rbln_device": 0},
            "vae": {"rbln_device": 0},
        },
    }


class TestSDMultiControlNetModel(BaseTest.TestModel):
    RBLN_CLASS = RBLNStableDiffusionControlNetPipeline
    HF_MODEL_ID = "hf-internal-testing/tiny-stable-diffusion-torch"
    CONTROLNET_ID = "hf-internal-testing/tiny-controlnet"

    GENERATION_KWARGS = {
        "prompt": "the mona lisa",
        "image": [
            torch.randn(1, 3, 64, 64, generator=torch.manual_seed(42)),
            torch.randn(1, 3, 64, 64, generator=torch.manual_seed(42)),
        ],
        "controlnet_conditioning_scale": [1.0, 0.8],
        "negative_prompt": "monochrome, lowres, bad anatomy, worst quality, low quality",
    }
    RBLN_CLASS_KWARGS = {
        "rbln_img_width": 64,
        "rbln_img_height": 64,
    }

    @classmethod
    def setUpClass(cls):
        controlnet = ControlNetModel.from_pretrained(cls.CONTROLNET_ID)
        controlnet_1 = ControlNetModel.from_pretrained(cls.CONTROLNET_ID)
        controlnets = [controlnet, controlnet_1]
        cls.RBLN_CLASS_KWARGS["controlnet"] = controlnets
        return super().setUpClass()


class TestKandinskyV22InpaintingModel(BaseTest.TestModel):
    RBLN_CLASS = RBLNKandinskyV22InpaintCombinedPipeline
    # HF_MODEL_ID = "hf-internal-testing/tiny-random-kandinsky-v22-decoder"
    HF_MODEL_ID = "kandinsky-community/kandinsky-2-2-decoder-inpaint"
    GENERATION_KWARGS = {
        "prompt": "concept art digital painting of an elven castle, inspired by lord of the rings, highly detailed, 8k",
        "generator": torch.manual_seed(42),
        "image": torch.FloatTensor(1, 3, 512, 512).uniform_(-1, 1),
        "mask_image": torch.randn(1, 1, 512, 512).uniform_(0, 1),
    }
    RBLN_CLASS_KWARGS = {
        "rbln_img_width": 512,
        "rbln_img_height": 512,
    }


class TestSVDImg2VidModel(BaseTest.TestModel):
    RBLN_CLASS = RBLNStableVideoDiffusionPipeline
    HF_MODEL_ID = "stabilityai/stable-video-diffusion-img2vid"

    GENERATION_KWARGS = {
        "num_inference_steps": 2,
        "generator": torch.manual_seed(42),
        "image": torch.randn(1, 3, 32, 32, generator=torch.manual_seed(42)).uniform_(0, 1),
        "num_frames": 2,
        "decode_chunk_size": 2,
        "output_type": "pt",
        "height": 32,
        "width": 32,
    }
    RBLN_CLASS_KWARGS = {
        "rbln_img_width": 32,
        "rbln_img_height": 32,
        "rbln_num_frames": 2,
        "rbln_decode_chunk_size": 2,
        "rbln_config": {
            "image_encoder": {"rbln_device": 0},
            "unet": {"rbln_device": 0},
            "vae": {"rbln_device": -1},
        },
    }

    @classmethod
    def get_dummy_components(cls):
        from diffusers import (
            AutoencoderKLTemporalDecoder,
            EulerDiscreteScheduler,
            UNetSpatioTemporalConditionModel,
        )
        from transformers import (
            CLIPImageProcessor,
            CLIPVisionConfig,
            CLIPVisionModelWithProjection,
        )

        torch.manual_seed(42)
        unet = UNetSpatioTemporalConditionModel(
            block_out_channels=(32, 64),
            layers_per_block=2,
            sample_size=32,
            in_channels=8,
            out_channels=4,
            down_block_types=(
                "CrossAttnDownBlockSpatioTemporal",
                "DownBlockSpatioTemporal",
            ),
            up_block_types=("UpBlockSpatioTemporal", "CrossAttnUpBlockSpatioTemporal"),
            cross_attention_dim=32,
            num_attention_heads=8,
            projection_class_embeddings_input_dim=96,
            addition_time_embed_dim=32,
        )
        scheduler = EulerDiscreteScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            interpolation_type="linear",
            num_train_timesteps=1000,
            prediction_type="v_prediction",
            sigma_max=700.0,
            sigma_min=0.002,
            steps_offset=1,
            timestep_spacing="leading",
            timestep_type="continuous",
            trained_betas=None,
            use_karras_sigmas=True,
        )

        torch.manual_seed(42)
        vae = AutoencoderKLTemporalDecoder(
            block_out_channels=[32, 64],
            in_channels=3,
            out_channels=3,
            down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D"],
            latent_channels=4,
        )

        torch.manual_seed(42)
        config = CLIPVisionConfig(
            hidden_size=32,
            projection_dim=32,
            num_hidden_layers=5,
            num_attention_heads=4,
            image_size=32,
            intermediate_size=37,
            patch_size=1,
        )
        image_encoder = CLIPVisionModelWithProjection(config)

        torch.manual_seed(42)
        feature_extractor = CLIPImageProcessor(crop_size=32, size=32)
        components = {
            "unet": unet,
            "image_encoder": image_encoder,
            "scheduler": scheduler,
            "vae": vae,
            "feature_extractor": feature_extractor,
        }
        return components

    @classmethod
    def setUpClass(cls):
        import os
        import shutil

        from .test_base import TestLevel

        env_coverage = os.environ.get("OPTIMUM_RBLN_TEST_LEVEL", "default")
        env_coverage = TestLevel[env_coverage.upper()]
        if env_coverage.value < cls.TEST_LEVEL.value:
            raise unittest.SkipTest(f"Skipped test : Test Coverage {env_coverage.name} < {cls.TEST_LEVEL.name}")

        if os.path.exists(cls.get_rbln_local_dir()):
            shutil.rmtree(cls.get_rbln_local_dir())

        components = cls.get_dummy_components()
        cls.model = cls.RBLN_CLASS.from_pretrained(
            cls.HF_MODEL_ID,
            export=True,
            model_save_dir=cls.get_rbln_local_dir(),
            rbln_device=cls.DEVICE,
            **cls.RBLN_CLASS_KWARGS,
            **cls.HF_CONFIG_KWARGS,
            **components,
        )
        # return super().setUpClass()

    def test_generate(self):
        inputs = self.get_inputs()
        output = self.model(**inputs).frames[0]
        output = self.postprocess(inputs, output)
        if self.EXPECTED_OUTPUT:
            self.assertEqual(output, self.EXPECTED_OUTPUT)


if __name__ == "__main__":
    unittest.main()
