import unittest

import torch
from diffusers import ControlNetModel

from optimum.rbln import (
    RBLNKandinskyV22CombinedPipeline,
    RBLNStableDiffusion3Img2ImgPipeline,
    RBLNStableDiffusion3Pipeline,
    RBLNStableDiffusionControlNetPipeline,
    RBLNStableDiffusionImg2ImgPipeline,
    RBLNStableDiffusionPipeline,
    RBLNStableDiffusionXLControlNetPipeline,
    RBLNStableDiffusionXLPipeline,
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
            "unet": {"batch_size": 2},
            "vae": {
                "sample_size": (64, 64),
            },
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
            },
            "unet": {
                "batch_size": 2,
            },
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
        "rbln_config": {
            "controlnet": {
                "batch_size": 2,
            },
            "unet": {
                "batch_size": 2,
            },
        },
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
        "rbln_config": {
            "unet": {
                "batch_size": 2,
            },
            "controlnet": {
                "batch_size": 2,
            },
        },
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
            "transformer": {
                "batch_size": 2,
            }
        }
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
        "rbln_config": {
            "image_size": (64, 64),
            "transformer": {"batch_size": 2},
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
        "rbln_config": {
            "controlnet": {
                "batch_size": 2,
            },
            "unet": {
                "batch_size": 2,
            },
        },
    }

    @classmethod
    def setUpClass(cls):
        controlnet = ControlNetModel.from_pretrained(cls.CONTROLNET_ID)
        controlnet_1 = ControlNetModel.from_pretrained(cls.CONTROLNET_ID)
        controlnets = [controlnet, controlnet_1]
        cls.RBLN_CLASS_KWARGS["controlnet"] = controlnets
        return super().setUpClass()


class TestKandinskyV22Model(BaseTest.TestModel):
    RBLN_CLASS = RBLNKandinskyV22CombinedPipeline
    HF_MODEL_ID = "hf-internal-testing/tiny-random-kandinsky-v22-decoder"
    GENERATION_KWARGS = {
        "prompt": "red cat, 4k photo",
        "generator": torch.manual_seed(42),
        "num_inference_steps": 3,
    }
    RBLN_CLASS_KWARGS = {
        "rbln_img_width": 64,
        "rbln_img_height": 64,
        "rbln_config": {
            "prior_pipe": {"prior": {"batch_size": 2}},
            "decoder_pipe": {"unet": {"batch_size": 2}},
        },
    }

    def test_complicate_config(self):
        rbln_config = {
            "prior_pipe": {
                "text_encoder": {
                    "batch_size": 2,
                },
            },
            "prior_prior": {
                "batch_size": 4,
            },
            "unet": {
                "batch_size": 2,
            },
            "batch_size": 1,
        }
        rbln_class_kwargs_copy = self.RBLN_CLASS_KWARGS.copy()
        rbln_class_kwargs_copy["rbln_config"] = rbln_config
        with self.subTest():
            _ = self.RBLN_CLASS.from_pretrained(
                model_id=self.HF_MODEL_ID,
                export=True,
                **rbln_class_kwargs_copy,
            )
        with self.subTest():
            self.assertEqual(_.prior_text_encoder.rbln_config.batch_size, 2)
            self.assertEqual(_.prior_prior.rbln_config.batch_size, 4)
            self.assertEqual(_.unet.rbln_config.batch_size, 2)


if __name__ == "__main__":
    unittest.main()
