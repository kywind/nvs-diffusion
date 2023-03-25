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
from kornia.geometry.depth import depth_to_3d, project_points, DepthWarper
from kornia.geometry.conversions import normalize_pixel_coordinates
from kornia.geometry.linalg import compose_transformations, \
        convert_points_to_homogeneous, inverse_transformation, transform_points

from modules.midas.model import MiDaS
from data.re10k_simple import RE10KDataset
from utils import PIL_utils


def warp(image_src, depth_src, pose_src, pose_tgt, camera_matrix, normalize_points=False):
    """
    camera_matrix (Tensor): tensor containing the camera intrinsics with shape (B, 3, 3).
    """

    # unproject source points to camera frame
    points_3d_src = depth_to_3d(depth_src, camera_matrix, normalize_points)  # Bx3xHxW

    # transform points from source to destination
    points_3d_src = points_3d_src.permute(0, 2, 3, 1)  # BxHxWx3

    # apply transformation to the 3d points
    trans = pose_tgt @ torch.linalg.inv(pose_src)
    points_3d_dst = transform_points(trans[:, None], points_3d_src)  # BxHxWx3

    # project back to pixels
    camera_matrix_tmp = camera_matrix[:, None, None]  # Bx1x1xHxW
    points_2d_dst = project_points(points_3d_dst, camera_matrix_tmp)  # BxHxWx2
    height, width = depth_src.shape[-2:]
    points_2d_dst[:, 0] = torch.clamp(points_2d_dst[:,0], 0, width-1)
    points_2d_dst[:, 1] = torch.clamp(points_2d_dst[:,1], 0, height-1)
    # points_2d_dst = points_2d_dst.long().numpy()
    
    # mask = np.zeros(depth_src.shape)
    # image = np.zeros(depth_src.shape)
    image_src = torch.tensor(np.array(image_src)).to(torch.float32).permute(2,0,1)[None]
    # mask_src = torch.ones_like(image_src)[:, 0:1]
    image_tgt = torch.ones_like(image_src)
    mask_tgt = torch.ones_like(image_src)[:, 0:1]

    # return points_2d_dst

    # normalize points between [-1 / 1]
    # points_2d_dst_norm = normalize_pixel_coordinates(points_2d_dst, height, width)  # BxHxWx2
    # import ipdb; ipdb.set_trace()

    # image_tgt = F.grid_sample(image_src, points_2d_dst, align_corners=True)
    # mask_tgt = F.grid_sample(mask_src, points_2d_dst, align_corners=True)

    for i in range(height):
        for j in range(width):
            x, y = points_2d_dst[0, i, j]
            x, y = int(x), int(y)
            if x < 0 or y < 0 or x >= width or y >= height:
                continue
            image_tgt[0, :, y-1:y+1, x-1:x+1] = image_src[0, :, i:i+1, j:j+1]
            mask_tgt[0, :, y-1:y+1, x-1:x+1] = 0
    
    # transform to PIL
    image_tgt = Image.fromarray(image_tgt[0].numpy().transpose(1,2,0).astype(np.uint8))
    mask_tgt = Image.fromarray((mask_tgt[0, 0].numpy() * 255).astype(np.uint8))
    
    # image_src = Image.fromarray(image_src[0].numpy().transpose(1,2,0).astype(np.uint8))
    # image_save = PIL_utils.concat(image_src, image_tgt)
    # mask_tgt.save('test.png')
    # import ipdb; ipdb.set_trace()

    return image_tgt, mask_tgt


def generate_video(dataset, seq_idx, depth_model, pipe, camera_traj=None, loop=True, test=False):
    inpaint_image = None  # reset the inpaint image
    image_save_list = []
    loop_len = dataset.get_seq_len(seq_idx)-1 if not test else 5
    for i in range(loop_len):
        example = dataset.get_item(seq_idx, i)
        example_next = dataset.get_item(seq_idx, i+5)

        if i == 0:
            image_src = example["image"]
            image_src = (image_src + 1.) * 127.5
            image_src = image_src.numpy().astype(np.uint8)
            image_src = Image.fromarray(image_src).resize((512, 512))
        else:
            assert inpaint_image is not None
            image_src = inpaint_image

        K = example["K"][None]
        Rt = example["Rt"][None]
        # Rt_next = Rt.clone()
        # Rt_next[0, 1, 3] -= 0.2
        # Rt_next[0, 2, 3] += 0.2
        Rt_next = example_next["Rt"][None]

        Rt[:, 0, 3] *= 512
        Rt[:, 1, 3] *= 512

        K[:, 0, 0] *= 512 / 256
        K[:, 1, 1] *= 512 / 256
        K[:, 0, 2] *= 512 / 256
        K[:, 1, 2] *= 512 / 256
        # image_next = example_next["image"]
        
        # compute depth
        disparity = depth_model.forward_PIL(image_src)
        disparity = cv2.resize(disparity, (512, 512), interpolation=cv2.INTER_LINEAR)
        disparity = torch.from_numpy(disparity)[None, None]
        # depth = baseline * focal / disparity
        depth = 100 / (disparity + 1e-4)
        # import ipdb; ipdb.set_trace()

        # warp image_src to the target pose by depth
        image_tgt, mask = warp(image_src, depth, Rt, Rt_next, K)

        # generate inpainted image
        prompt = "Real estate photo"
        inpaint_image = pipe(prompt=prompt, image=image_src, mask_image=mask).images[0]
        inpaint_image = PIL_utils.blend(inpaint_image, image_tgt, mask)

        # visualize masked image
        # image_src_numpy = np.array(image_src)
        # mask_numpy = np.array(mask)[:,:,None]
        # masked_image = Image.fromarray(image_src_numpy * (mask_numpy == 0))
        
        # concat image for saving
        image_save = PIL_utils.concat(image_tgt, inpaint_image)
        image_save = PIL_utils.concat(image_src, image_save)
        image_save_list.append(image_save)

        if loop:
            image_src = inpaint_image  # use the inpainted image as the next input
    
    # return video
    return image_save_list


if __name__ == "__main__":
    device = "cuda"
    rootPath = 'vis/0319-depth-warp/'
    os.makedirs(rootPath, exist_ok=True)

    dataset = RE10KDataset(task='val', size=256)

    pipe = StableDiffusionInpaintPipeline.from_pretrained("stabilityai/stable-diffusion-2-inpainting", torch_dtype=torch.float16)
    pipe = pipe.to(device)

    model_weights = "checkpoints/dpt_large-midas-2f21e586.pt"
    model_type = "dpt_large"
    depth_model = MiDaS(model_weights, model_type, device=device, optimize=False)

    for i in range(len(dataset)):
        outPath = os.path.join(rootPath, f'{i}/')
        os.makedirs(outPath, exist_ok=True)

        video = generate_video(dataset, i, depth_model, pipe, loop=True, test=True)

        for img_idx, image in enumerate(video):
            image.save(os.path.join(outPath, f'{img_idx}.png'))
