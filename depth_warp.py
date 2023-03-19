import argparse
import numpy as np
import cv2
import math
import os
import torch
import torch.nn.functional as F
import PIL
import requests
from io import BytesIO
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from diffusers import StableDiffusionInpaintPipeline

import kornia
from kornia.geometry.depth import depth_from_disparity, \
        depth_to_3d, project_points, DepthWarper
from kornia.geometry.conversions import normalize_pixel_coordinates
from kornia.geometry.linalg import compose_transformations, \
        convert_points_to_homogeneous, inverse_transformation, transform_points

from modules.midas.model import MiDaS
from data.re10k import RE10KValidation
from utils import PIL_utils


def warp(image_src, depth_src, dst_trans_src, camera_matrix, normalize_points=False):
    # unproject source points to camera frame
    points_3d_src = depth_to_3d(depth_src, camera_matrix, normalize_points)  # Bx3xHxW

    # transform points from source to destination
    points_3d_src = points_3d_src.permute(0, 2, 3, 1)  # BxHxWx3

    # apply transformation to the 3d points
    points_3d_dst = transform_points(dst_trans_src[:, None], points_3d_src)  # BxHxWx3

    # project back to pixels
    camera_matrix_tmp = camera_matrix[:, None, None]  # Bx1x1xHxW
    points_2d_dst = project_points(points_3d_dst, camera_matrix_tmp)  # BxHxWx2
    height, width = depth_src.shape[-2:]
    points_2d_dst[:, 0] = torch.clamp(points_2d_dst[:,0], 0, width-1)
    points_2d_dst[:, 1] = torch.clamp(points_2d_dst[:,1], 0, height-1)
    points_2d_dst = points_2d_dst.long().numpy()
    
    mask = np.zeros(depth_src.shape)
    image = np.zeros(depth_src.shape)

    import ipdb; ipdb.set_trace()
    return points_2d_dst

    # # normalize points between [-1 / 1]
    # points_2d_src_norm = normalize_pixel_coordinates(points_2d_dst, height, width)  # BxHxWx2
    # return F.grid_sample(image_src, points_2d_src_norm, align_corners=True)


if __name__ == "__main__":
    device = "cuda"
    rootPath = 'vis/0319-depth-warp/'
    os.makedirs(rootPath, exist_ok=True)

    dataset = RE10KValidation(size=256, low=1, high=30, interval=10)
    dl = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=True, num_workers=0)

    pipe = StableDiffusionInpaintPipeline.from_pretrained("stabilityai/stable-diffusion-2-inpainting", torch_dtype=torch.float16)
    pipe = pipe.to(device)
    
    model_weights = "checkpoints/dpt_large-midas-2f21e586.pt"
    model_type = "dpt_large"
    depth_model = MiDaS(model_weights, model_type, device=device, optimize=False)

    seq_cnt = 0  # sequence count
    for idx, batch in enumerate(dl):
        seq_idx, img_idx = batch["index"][0, 0].item(), batch["index"][0, 1].item()
        seq_idx, img_idx = int(seq_idx), int(img_idx)
        
        if seq_idx >= seq_cnt:
            outPath = os.path.join(rootPath, f'{seq_cnt}/')
            os.makedirs(outPath, exist_ok=True)
            seq_cnt += 1
            init_image = batch["image"][0]
            init_image = (init_image + 1.) * 127.5
            init_image = init_image.numpy()
            init_image = Image.fromarray(init_image.astype(np.uint8)).resize((512, 512))

        camera = batch["camera"]
        disparity = depth_model.forward_PIL(init_image)
        warp_image, mask_image = warp(init_image, camera, disparity)

        prompt = "Real estate photo"
        inpaint_image = pipe(prompt=prompt, image=init_image, mask_image=mask_image).images[0]
        
        inpaint_image = PIL_utils.blend(inpaint_image, warp_image, mask_image)

        init_image_numpy = np.array(init_image)
        mask_image_numpy = np.array(mask_image)
        masked_image = Image.fromarray(init_image_numpy * (mask_image_numpy == 0))
        
        image_save = PIL_utils.concat(masked_image, inpaint_image)
        image_save = PIL_utils.concat(init_image, image_save)
        image_save.save(os.path.join(outPath, f'{img_idx}.png'))

        if loop:
            init_image = inpaint_image  # use the inpainted image as the next input
