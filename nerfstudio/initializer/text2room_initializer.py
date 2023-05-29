"""
Code for inpainting with diffusion.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type, Union


import torch
from torchtyping import TensorType

from nerfstudio.initializer.base_initializer import Initializer

from nerfstudio.inpainter.base_inpainter import Inpainter
from diffusers import StableDiffusionImg2ImgPipeline
from diffusers import DiffusionPipeline

from nerfstudio.configs.base_config import InstantiateConfig

from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.cameras.rays import RayBundle

import kornia
from kornia.geometry.depth import depth_to_3d, project_points, DepthWarper
from kornia.geometry.conversions import normalize_pixel_coordinates
from kornia.geometry.linalg import compose_transformations, \
        convert_points_to_homogeneous, inverse_transformation, transform_points

from PIL import Image
import cv2
import numpy as np



@dataclass
class Text2RoomInitializer(Initializer):
    """Model class for inpainting.

    Args:
        config: configuration for instantiating.
    """

    def __init__(
        self,
        device: str = "cuda",
        **kwargs
    ) -> None:
        super().__init__(**kwargs)

        self.device = device



    def initialize_scene(
        self,
        train_dataset: InputDataset,
        inpainter: Optional[Inpainter] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Forward pass for inpainting.

        Args:
            images: input images.
            mask: mask for inpainting.

        Returns:
            outputs: outputs from the inpainting model.
        """

        if "prompt" in kwargs:
            prompt = kwargs["prompt"]
        else:
            prompt = "Real estate photo"

        if self.model_type == 'stable_diffusion':
            image = (image + 1.) * 127.5
            image = image.detach().cpu().numpy().astype(np.uint8)
            image = Image.fromarray(image).resize((512, 512))

            # import ipdb; ipdb.set_trace()
            # mask = mask.detach().cpu().numpy()[..., 0]
            # mask = np.floor(mask) * 255
            # mask = mask.astype(np.uint8)
            # mask = Image.fromarray(mask).resize((512, 512))
            # mask.save(f"temp/vis-temp/mask_{step}.png")

            inpaint_image = self.model(
                prompt=prompt, 
                image=image, 
                strength=0.5,
                guidance_scale=7.5,  # default value
            ).images[0]
            # inpaint_image = torch.from_numpy(np.array(inpaint_image)).to(self.device).permute(2, 0, 1).float() / 127.5 - 1.
        
        elif self.model_type == 'floyd':

            prompt_embeds, negative_embeds = self.stage_1.encode_prompt(prompt)

            generator = torch.manual_seed(0)

            image = self.stage_1(prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_embeds, generator=generator, output_type="pt").images
            # pt_to_pil(image)[0].save("./if_stage_I.png")

            # stage 2
            image = self.stage_2(
                image=image, prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_embeds, generator=generator, output_type="pt"
            ).images
            # pt_to_pil(image)[0].save("./if_stage_II.png")

            # stage 3
            image = self.stage_3(prompt=prompt, image=image, generator=generator, noise_level=100).images
            # image[0].save("./if_stage_III.png")
            import ipdb; ipdb.set_trace()

            inpaint_image = image[0]

        return inpaint_image
