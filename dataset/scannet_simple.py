import os
import numpy as np
import pickle
import cv2
from PIL import Image
from torch.utils.data import Dataset
import torch
import torchvision
from torchvision.transforms import Compose, Normalize, Resize, ToTensor
import glob


class ScanNetDataset(Dataset):
    def __init__(self, datapath='data/scannet_minival/'):
        super(ScanNetDataset, self).__init__()
        self.datapath = datapath
        self.img_size = [512, 512]  # cfg.TRAIN.IMG_SIZE
        self.ori_size = [480, 640]
        self.depth_min = 0.25  # cfg.TRAIN.MIN_DEPTH
        self.depth_max = 20  # cfg.TRAIN.MAX_DEPTH
        self.depth_range = True  # cfg.TRAIN.DEPTH_RANGE
        self.depth_level = 64  # cfg.TRAIN.DEPTH_LEVEL
        self.IMG_EXT = ['png', 'jpg']

        self.image_path_list = []
        self.depth_path_list = []
        self.pose_list = []
        self.intr_image_list = []
        self.intr_depth_list = []

        self.video_paths = sorted(glob.glob(self.datapath + '*/'))
        for vid_path in self.video_paths: 
            self.image_path_list.append(
                sum([sorted(glob.glob(vid_path + f'frames/color/*.{ext}')) for ext in self.IMG_EXT],[])
            )
            self.depth_path_list.append(
                sum([sorted(glob.glob(vid_path + f'frames/depth/*.{ext}')) for ext in self.IMG_EXT],[])
            )
            pose_path_list = sorted(glob.glob(vid_path + f'frames/pose/*.txt'))
            pose_list = []
            for pose_path in pose_path_list:
                with open(pose_path, 'r') as f:
                    pose = f.read().split()
                    pose = np.array(pose).astype(np.float32).reshape(4, 4)
                pose_list.append(pose)
            self.pose_list.append(pose_list)

            with open(vid_path + 'frames/intrinsic/intrinsic_color.txt', 'r') as f:
                intr_image = f.read().split()
                intr_image = np.array(intr_image).astype(np.float32).reshape(4, 4)[:3, :3]
            self.intr_image_list.append(intr_image)

            with open(vid_path + 'frames/intrinsic/intrinsic_depth.txt', 'r') as f:
                intr_depth = f.read().split()
                intr_depth = np.array(intr_depth).astype(np.float32).reshape(4, 4)[:3, :3]
            self.intr_depth_list.append(intr_depth)

    def __len__(self):
        return len(self.metas)

    def read_img(self, filepath):
        img = Image.open(filepath)
        return img

    def read_depth(self, filepath):
        # Read depth image and camera pose
        depth_im = cv2.imread(filepath, -1).astype(
            np.float32)
        depth_im /= 1000.  # depth is saved in 16-bit PNG in millimeters
        depth_im[np.isinf(depth_im)] = 0
        depth_im[np.isnan(depth_im)] = 0

        assert np.max(depth_im) > self.depth_min
        mask = np.ones_like(depth_im)
        mask[depth_im < self.depth_min] = 0
        mask[depth_im > self.depth_max] = 0
        depth_im[depth_im < self.depth_min] = np.mean(depth_im)
        depth_im[depth_im > self.depth_max] = self.depth_max

        disp_im = 1. / depth_im

        return depth_im, disp_im, mask
    
    def update_intrinsics(self, K, ori_size, new_size):
        ori_height, ori_width = ori_size
        new_height, new_width = new_size

        factor_x = float(new_width) / float(ori_width)
        factor_y = float(new_height) / float(ori_height)

        fx = K[0, 0]
        fy = K[1, 1]
        cx = K[0, 2]
        cy = K[1, 2]

        fx *= factor_x
        fy *= factor_y
        cx *= factor_x
        cy *= factor_y

        return np.array([[fx, 0, cx],
                         [0, fy, cy],
                         [0, 0, 1]])

    def __len__(self):
        return len(self.image_path_list)

    def get_seq_len(self, seq_idx):
        return len(self.image_path_list[seq_idx])
    
    def resize(self, img, depth, intr, new_size=512):
        # h_d, w_d = depth.shape  # 480, 640
        h, w, _ = img.shape  # 968, 1296
        offset_x = (w - min(h, w)) // 2
        offset_y = (h - min(h, w)) // 2
        ratio = new_size / min(h, w)

        img = cv2.resize(img, (new_size, new_size), interpolation=cv2.INTER_LINEAR)
        depth = cv2.resize(depth, (new_size, new_size), interpolation=cv2.INTER_NEAREST)

        intr[0, 2] -= offset_x
        intr[1, 2] -= offset_y
        intr[:2, :] *= ratio
        # intrinsics = self.update_intrinsics(intr, (h, w), (new_size, new_size))
        return img, depth, intr
    
    def get_item(self, seq_idx, img_idx):

        img_path = self.image_path_list[seq_idx][img_idx]
        img = self.read_img(img_path)
        depth_path = self.depth_path_list[seq_idx][img_idx]
        depth, disp, mask = self.read_depth(depth_path)

        intr_img, intr_depth = self.intr_image_list[seq_idx], self.intr_depth_list[seq_idx]
        extrinsics = self.pose_list[seq_idx][img_idx]

        img = np.array(img) / 127. - 1.
        img, depth, intr = self.resize(img, depth, intr_img, new_size=512)

        # if self.depth_range:
        #     depth_values = np.array([max(np.min(depth), self.depth_min), \
        #                              min(np.max(depth), self.depth_max)], np.float32)
        # else:
        #     depth_values = np.array([self.depth_min, self.depth_max], np.float32)

        items = {
            'image': torch.tensor(img, dtype=torch.float32),# .permute(2, 0, 1),
            'depth': torch.tensor(depth, dtype=torch.float32),
            'K': torch.tensor(intr, dtype=torch.float32),
            'Rt': torch.tensor(extrinsics, dtype=torch.float32),
        }

        return items


if __name__ == '__main__':
    ds = ScanNetDataset()
    for i in range(len(ds)):
        fl = ds.get_seq_len(i)
        for j in range(fl):
            ds.get_item(i, j)