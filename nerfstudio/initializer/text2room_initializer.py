"""
Code for inpainting with diffusion.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type, Union


import torch
from torchtyping import TensorType

from nerfstudio.initializer.base_initializer import Initializer, InitializerConfig

from nerfstudio.inpainter.base_inpainter import Inpainter
from diffusers import StableDiffusionImg2ImgPipeline
from diffusers import DiffusionPipeline

from nerfstudio.configs.base_config import InstantiateConfig

from nerfstudio.data.datamanagers.base_datamanager import DataManager
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
import os
import json

from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline, StableDiffusionInpaintPipeline



@dataclass
class Text2RoomInitializerConfig(InitializerConfig):
    """Text2Room dataset config"""

    _target: Type = field(default_factory=lambda: Text2RoomInitializer)
    """target class to instantiate"""


class Text2RoomInitializer(Initializer):
    """Model class for inpainting.

    Args:
        config: configuration for instantiating.
    """

    def __init__(
        self,
        config: Text2RoomInitializerConfig, 
        initialize_save_dir: str = None, 
        device: str = "cuda",
    ) -> None:
        super().__init__()

        self.config = config
        self.device = device
        self.initialize_save_dir = initialize_save_dir


    def initialize_scene(
        self,
        **kwargs
    ) -> Dict[str, Any]:
        """Forward pass for inpainting.

        Args:
            images: input images.
            mask: mask for inpainting.

        Returns:
            outputs: outputs from the inpainting model.
        """

        frames = []

        if "prompt" in kwargs:
            prompt = kwargs["prompt"]
        else:
            prompt = " living room with a lit furnace, couch and cozy curtains, bright lamps that make the room look well-lit"

        model_path = "stabilityai/stable-diffusion-2-1"
        pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        pipe = pipe.to(self.device)

        pipe.set_progress_bar_config(**{
            "leave": False,
            "desc": "Generating Start Image"
        })

        init_image = pipe(prompt).images[0]
        save_path = os.path.join(self.initialize_save_dir, '00000.png')
        init_image.save(save_path)

        frames.append(
            {
                'file_path': save_path,
                'transform_matrix': [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0]
                ]
            }
        )

        json_save_path = os.path.join(self.initialize_save_dir, 'transforms.json')
        json_file = {}

        json_file['fl_x'] = 768.0
        json_file['fl_y'] = 768.0
        json_file['cx'] = 384.0
        json_file['cy'] = 384.0
        json_file['w'] = 768
        json_file['h'] = 768
        json_file['frames'] = frames

        with open(json_save_path, 'w') as f:
                json.dump(json_file, f)

    def set_trajectory(self, trajectory: List[Dict[str, Any]]):
        pass
