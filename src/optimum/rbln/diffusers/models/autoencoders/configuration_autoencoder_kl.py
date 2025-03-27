from typing import Optional, Tuple

from ....configuration_utils import RBLNModelConfig


class RBLNAutoencoderKLConfig(RBLNModelConfig):
    def __init__(
        self,
        batch_size: Optional[int] = None,
        sample_size: Optional[Tuple[int, int]] = None,
        image_size: Optional[Tuple[int, int]] = None,
        img_height: Optional[int] = None,
        img_width: Optional[int] = None,
        img2img_pipeline: Optional[bool] = None,
        inpaint_pipeline: Optional[bool] = None,
        vae_scale_factor: Optional[float] = None,
        in_channels: Optional[int] = None,
        latent_channels: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.batch_size = batch_size or 1
        self.img2img_pipeline = img2img_pipeline
        self.inpaint_pipeline = inpaint_pipeline
        self.vae_scale_factor = vae_scale_factor
        self.in_channels = in_channels
        self.latent_channels = latent_channels

        if image_size is not None and (img_height is not None or img_width is not None):
            raise ValueError("image_size and img_height/img_width cannot both be provided")

        if img_height is not None and img_width is not None:
            self.image_size = (img_height, img_width)
        else:
            self.image_size = image_size

        if self.image_size is not None and sample_size is not None:
            raise ValueError("image_size and sample_size cannot both be provided")

        if self.image_size is None and sample_size is not None:
            self.image_size = sample_size

        if isinstance(self.image_size, int):
            self.image_size = (self.image_size, self.image_size)

    @property
    def sample_size(self):
        return self.image_size

    @property
    def latent_sample_size(self):
        return (self.image_size[0] // self.vae_scale_factor, self.image_size[1] // self.vae_scale_factor)

    @property
    def needs_encoder(self):
        return self.img2img_pipeline or self.inpaint_pipeline
