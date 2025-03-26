from typing import Optional

from ....configuration_utils import RBLNModelConfig


class RBLNAutoencoderKLConfig(RBLNModelConfig):
    def __init__(
        self,
        batch_size: Optional[int] = None,
        sample_size: Optional[int] = None,
        img_height: Optional[int] = None,
        img_width: Optional[int] = None,
        img2img_pipeline: Optional[bool] = None,
        inpaint_pipeline: Optional[bool] = None,
        vae_scale_factor: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.batch_size = batch_size or 1
        self.image_size = (img_height, img_width)
        self.sample_size = sample_size
        self.img2img_pipeline = img2img_pipeline
        self.inpaint_pipeline = inpaint_pipeline
        self.vae_scale_factor = vae_scale_factor

    @property
    def encoder_input_shape(self):
        return (self.sample_size[0], self.sample_size[1])

    @property
    def decoder_input_shape(self):
        return (self.sample_size[0] // self.vae_scale_factor, self.sample_size[1] // self.vae_scale_factor)

    @property
    def need_encoder(self):
        return self.img2img_pipeline or self.inpaint_pipeline
