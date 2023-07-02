import random
import numpy as np
from scipy.spatial.transform import Slerp, Rotation
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import trimesh

from nerfstudio.initializer.sds.utils.utils import get_rays, safe_normalize
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.configs.base_config import InstantiateConfig

DIR_COLORS = np.array([
    [255, 0, 0, 255], # front
    [0, 255, 0, 255], # side
    [0, 0, 255, 255], # back
    [255, 255, 0, 255], # side
    [255, 0, 255, 255], # overhead
    [0, 255, 255, 255], # bottom
], dtype=np.uint8)

def visualize_poses(poses, dirs, size=0.1):
    # poses: [B, 4, 4], dirs: [B]

    axes = trimesh.creation.axis(axis_length=4)
    sphere = trimesh.creation.icosphere(radius=1)
    objects = [axes, sphere]

    for pose, dir in zip(poses, dirs):
        # a camera is visualized with 8 line segments.
        pos = pose[:3, 3]
        a = pos + size * pose[:3, 0] + size * pose[:3, 1] - size * pose[:3, 2]
        b = pos - size * pose[:3, 0] + size * pose[:3, 1] - size * pose[:3, 2]
        c = pos - size * pose[:3, 0] - size * pose[:3, 1] - size * pose[:3, 2]
        d = pos + size * pose[:3, 0] - size * pose[:3, 1] - size * pose[:3, 2]

        segs = np.array([[pos, a], [pos, b], [pos, c], [pos, d], [a, b], [b, c], [c, d], [d, a]])
        segs = trimesh.load_path(segs)

        # different color for different dirs
        segs.colors = DIR_COLORS[[dir]].repeat(len(segs.entities), 0)

        objects.append(segs)

    trimesh.Scene(objects).show()

def get_view_direction(thetas, phis, overhead, front):
    #                   phis [B,];          thetas: [B,]
    # front = 0         [0, front)
    # side (right) = 1   [front, 180)
    # back = 2          [180, 180+front)
    # side (left) = 3  [180+front, 360)
    # top = 4                               [0, overhead]
    # bottom = 5                            [180-overhead, 180]
    res = torch.zeros(thetas.shape[0], dtype=torch.long)
    # first determine by phis
    phis = phis % (2 * np.pi)
    res[(phis < front / 2) | (phis >= 2 * np.pi - front / 2)] = 0
    res[(phis >= front / 2) & (phis < np.pi - front / 2)] = 1
    res[(phis >= np.pi - front / 2) & (phis < np.pi + front / 2)] = 2
    res[(phis >= np.pi + front / 2) & (phis < 2 * np.pi - front / 2)] = 3
    # override by thetas
    res[thetas <= overhead] = 4
    res[thetas >= (np.pi - overhead)] = 5
    return res


def rand_poses(size, device, radius_range=[1, 1.5], theta_range=[0, 120], phi_range=[0, 360], 
    return_dirs=False, angle_overhead=30, angle_front=60, jitter=False, uniform_sphere_rate=0.5):
    ''' generate random poses from an orbit camera
    Args:
        size: batch size of generated poses.
        device: where to allocate the output.
        radius: camera radius
        theta_range: [min, max], should be in [0, pi]
        phi_range: [min, max], should be in [0, 2 * pi]
    Return:
        poses: [size, 4, 4]
    '''

    theta_range = np.array(theta_range) / 180 * np.pi
    phi_range = np.array(phi_range) / 180 * np.pi
    angle_overhead = angle_overhead / 180 * np.pi
    angle_front = angle_front / 180 * np.pi

    radius = torch.rand(size, device=device) * (radius_range[1] - radius_range[0]) + radius_range[0]

    if random.random() < uniform_sphere_rate:
        unit_centers = F.normalize(
            torch.stack([
                (torch.rand(size, device=device) - 0.5) * 2.0,
                torch.rand(size, device=device),
                (torch.rand(size, device=device) - 0.5) * 2.0,
            ], dim=-1), p=2, dim=1
        )
        thetas = torch.acos(unit_centers[:,1])
        phis = torch.atan2(unit_centers[:,0], unit_centers[:,2])
        phis[phis < 0] += 2 * np.pi
        centers = unit_centers * radius.unsqueeze(-1)
    else:
        thetas = torch.rand(size, device=device) * (theta_range[1] - theta_range[0]) + theta_range[0]
        phis = torch.rand(size, device=device) * (phi_range[1] - phi_range[0]) + phi_range[0]
        phis[phis < 0] += 2 * np.pi

        centers = torch.stack([
            radius * torch.sin(thetas) * torch.sin(phis),
            radius * torch.cos(thetas),
            radius * torch.sin(thetas) * torch.cos(phis),
        ], dim=-1) # [B, 3]

    targets = 0

    # jitters
    if jitter:
        centers = centers + (torch.rand_like(centers) * 0.2 - 0.1)
        targets = targets + torch.randn_like(centers) * 0.2

    # lookat
    forward_vector = safe_normalize(centers - targets)
    up_vector = torch.FloatTensor([0, 1, 0]).to(device).unsqueeze(0).repeat(size, 1)
    right_vector = safe_normalize(torch.cross(forward_vector, up_vector, dim=-1))

    if jitter:
        up_noise = torch.randn_like(up_vector) * 0.02
    else:
        up_noise = 0

    up_vector = safe_normalize(torch.cross(right_vector, forward_vector, dim=-1) + up_noise)

    poses = torch.eye(4, dtype=torch.float, device=device).unsqueeze(0).repeat(size, 1, 1)
    poses[:, :3, :3] = torch.stack((right_vector, up_vector, forward_vector), dim=-1)
    poses[:, :3, 3] = centers

    if return_dirs:
        dirs = get_view_direction(thetas, phis, angle_overhead, angle_front)
    else:
        dirs = None

    # back to degree
    thetas = thetas / np.pi * 180
    phis = phis / np.pi * 180

    return poses, dirs, thetas, phis, radius


def circle_poses(device, radius=torch.tensor([3.2]), theta=torch.tensor([60]), phi=torch.tensor([0]), 
        return_dirs=False, angle_overhead=30, angle_front=60):

    theta = theta / 180 * np.pi
    phi = phi / 180 * np.pi
    angle_overhead = angle_overhead / 180 * np.pi
    angle_front = angle_front / 180 * np.pi

    centers = torch.stack([
        radius * torch.sin(theta) * torch.sin(phi),
        radius * torch.cos(theta),
        radius * torch.sin(theta) * torch.cos(phi),
    ], dim=-1) # [B, 3]

    # lookat
    forward_vector = safe_normalize(centers)
    up_vector = torch.FloatTensor([0, 1, 0]).to(device).unsqueeze(0).repeat(len(centers), 1)
    right_vector = safe_normalize(torch.cross(forward_vector, up_vector, dim=-1))
    up_vector = safe_normalize(torch.cross(right_vector, forward_vector, dim=-1))

    poses = torch.eye(4, dtype=torch.float, device=device).unsqueeze(0).repeat(len(centers), 1, 1)
    poses[:, :3, :3] = torch.stack((right_vector, up_vector, forward_vector), dim=-1)
    poses[:, :3, 3] = centers

    if return_dirs:
        dirs = get_view_direction(theta, phi, angle_overhead, angle_front)
    else:
        dirs = None

    return poses, dirs

@dataclass
class SDSDatasetConfig(InstantiateConfig):
    """Configuration for model instantiation"""

    _target: Type = field(default_factory=lambda: SDSDataset)
    """target class to instantiate"""
    height: int = 64
    """image height"""
    width: int = 64
    """image width"""

class SDSDataset:
    def __init__(
        self, 
        config,
        device: str = "cuda",
        max_iter: int = 10000,
    ):
        self.batch_size = 1
        self.min_near = 0.01
        self.angle_overhead = 30
        self.angle_front = 60
        self.jitter_pose = True
        self.uniform_sphere_rate = 0
        self.known_view_scale = 1.5

        self.default_radius = 3.2
        self.default_fovy = 20
        self.default_polar = 90
        self.default_azimuth = 0

        self.fovy_range = [10, 30]
        self.radius_range = [0.5, 1.5]
        self.theta_range = [45, 105]
        self.phi_range = [-180, 180]

        self.device = device

        self.H = config.height
        self.W = config.width
        self.max_iter = max_iter

        # self.mode = mode # train, val, test
        # self.training = self.mode in ['train', 'all']

        self.cx = self.H / 2
        self.cy = self.W / 2

        self.near = self.min_near
        self.far = 1000 # infinite

        ## visualize poses
        # poses, dirs, _, _, _ = rand_poses(100, self.device, 
        #     radius_range=self.radius_range, angle_overhead=self.angle_overhead, 
        #     angle_front=self.angle_front, jitter=self.jitter_pose, uniform_sphere_rate=1)
        # visualize_poses(poses.detach().cpu().numpy(), dirs.detach().cpu().numpy())


    def collate(self, index):
        B = len(index) # always 1
        # if self.training:
        if True:
            # random pose on the fly
            poses, dirs, thetas, phis, radius = rand_poses(B, self.device, 
                radius_range=self.radius_range, theta_range=self.theta_range, phi_range=self.phi_range, 
                return_dirs=True, angle_overhead=self.angle_overhead, angle_front=self.angle_front, 
                jitter=self.jitter_pose, uniform_sphere_rate=self.uniform_sphere_rate)

            # random focal
            fov = random.random() * (self.fovy_range[1] - self.fovy_range[0]) + self.fovy_range[0]
        # else:
        #     # circle pose
        #     thetas = torch.FloatTensor([self.default_polar]).to(self.device)
        #     phis = torch.FloatTensor([(index[0] / self.size) * 360]).to(self.device)
        #     radius = torch.FloatTensor([self.default_radius]).to(self.device)
        #     poses, dirs = circle_poses(self.device, radius=radius, theta=thetas, phi=phis, return_dirs=True, 
        #         angle_overhead=self.angle_overhead, angle_front=self.angle_front)
        #     # fixed focal
        #     fov = self.default_fovy

        focal = self.H / (2 * np.tan(np.deg2rad(fov) / 2))
        intrinsics = np.array([focal, focal, self.cx, self.cy])

        projection = torch.tensor([
            [2*focal/self.W, 0, 0, 0],
            [0, -2*focal/self.H, 0, 0],
            [0, 0, -(self.far+self.near)/(self.far-self.near), -(2*self.far*self.near)/(self.far-self.near)],
            [0, 0, -1, 0]
        ], dtype=torch.float32, device=self.device).unsqueeze(0)
        # import ipdb; ipdb.set_trace()

        mvp = projection @ torch.inverse(poses) # [1, 4, 4]

        # sample a low-resolution but full image
        rays = get_rays(poses, intrinsics, self.H, self.W, -1)
        fx, fy, cx, cy = intrinsics

        # delta polar/azimuth/radius to default view
        delta_polar = thetas - self.default_polar
        delta_azimuth = phis - self.default_azimuth
        delta_azimuth[delta_azimuth > 180] -= 360 # range in [-180, 180]
        delta_radius = radius - self.default_radius

        # distortion_params = torch.stack(distort, dim=0)[idx_tensor]

        cameras = Cameras(
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            height=self.H,
            width=self.W,
            camera_to_worlds=poses[:, :3, :4],
            camera_type=CameraType.PERSPECTIVE,
        )
        # import ipdb; ipdb.set_trace()
        # c = ray_indices[:, 0]  # camera indices
        # y = ray_indices[:, 1]  # row indices
        # x = ray_indices[:, 2]  # col indices
        # coords = image_coords[y, x]
        image_coords = cameras.get_image_coords()# .to(cameras.device)
        # pose_optimizer: CameraOptimizer
        # camera_opt_to_camera = self.pose_optimizer(c)
        ray_bundle = cameras.generate_rays(
            camera_indices=0,
            coords=image_coords.reshape(-1, 2),
        )
        # import ipdb; ipdb.set_trace()
        return {
            'ray_bundle': ray_bundle,
            'H': self.H,
            'W': self.W,
            'polar': delta_polar,
            'azimuth': delta_azimuth,
            'radius': delta_radius,
        }
        # import ipdb; ipdb.set_trace()

        # data = {
        #     'H': self.H,
        #     'W': self.W,
        #     'rays_o': rays['rays_o'],
        #     'rays_d': rays['rays_d'],
        #     'dir': dirs,
        #     'mvp': mvp,
        #     'polar': delta_polar,
        #     'azimuth': delta_azimuth,
        #     'radius': delta_radius,
        # }
        # return data

    def dataloader(self, batch_size=None):
        batch_size = batch_size or self.batch_size
        loader = DataLoader(list(range(self.max_iter)), batch_size=batch_size, 
            collate_fn=self.collate, shuffle=True, num_workers=0)
            # collate_fn=self.collate, shuffle=self.training, num_workers=0)
        loader._data = self
        return iter(loader)