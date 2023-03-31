import os
import random
import torch
import numpy as np
import PIL
import glob
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm


class RE10KDataset(Dataset):
    def __init__(self, task='train', 
                       size=256):
        self.task = task
        if self.task == 'train':
            self.data_path = '/data/yangfu2/data/re10k/train/'
            self.camera_dir = '/data/yangfu2/data/re10k/anns/train/'
        elif self.task == 'val':
            self.data_path = '/data/yangfu2/data/re10k/sample/'
            self.camera_dir = '/data/yangfu2/data/re10k/anns/sample/'
        else:
            assert self.task == 'test'
            self.data_path = '/data/yangfu2/data/re10k/test/'
            self.camera_dir = '/data/yangfu2/data/re10k/anns/test/'

        self.IMG_EXT = ['png', 'jpg']
        self.image_size = size
        if self.data_path is None:
            raise RuntimeError(f"{task} dataset not exist.")
        self.video_paths = sorted(glob.glob(self.data_path + '/*/'))

        print('loading data...')
        self.image_list = []
        self.camera_list = []
        for idx in tqdm(range(len(self.video_paths))):
            frames = sum([sorted(glob.glob(self.video_paths[idx] + f'/*.{ext}')) for ext in self.IMG_EXT],[])
            if len(frames) == 0: continue
            self.image_list.append(frames)
            self.camera_list.append(self.read_camera(frames))

    
    def read_camera(self, frames):
        assert len(frames) > 0
        seq_id = os.path.split(os.path.split(frames[0])[0])[-1]
        cam = np.loadtxt(os.path.join(self.camera_dir, seq_id +".txt"), skiprows=0, comments="http").reshape(-1, 19)
        cam_dict = {
            'K': None,
            'K_inv': None,
            'R': [],
            't': [],
        }
        for name in frames:
            img_id = os.path.split(name)[-1].split(".")[0]

            intr = cam[np.where(int(img_id)==cam)[0][0]][1:7]
            extr = cam[np.where(int(img_id)==cam)[0][0]][7:]

            ## rescale intrinsics to pixel level (is it needed)?
            rescale = self.image_size
            intr = intr * rescale
            # norm_factor = self.image_size / 2
            # f_ndc = np.array([intr[0] / norm_factor, intr[1] / norm_factor])
            # pp_ndc = np.array(
            #     [(intr[2] - self.image_size/2) / norm_factor, 
            #      (intr[3] - self.image_size/2) / norm_factor])

            intr = np.array(
                    [
                        [intr[0], 0, intr[2]],
                        [0, intr[1], intr[3]],
                        [0, 0, 1],
                    ],
                    dtype=np.float32,
                )
            if cam_dict['K'] is None:
                cam_dict['K'] = intr
                # cam_dict['K_inv'] = np.linalg.inv(intr)
            else:
                assert abs((cam_dict['K'] - intr).sum()) < 1e-5

            extr = extr.reshape(3, 4)
            R = extr[:, :3].astype(np.float32)  # 3,3
            t = extr[:, 3].astype(np.float32)  # 3,

            cam_dict['R'].append(R)
            cam_dict['t'].append(t)
            
        return cam_dict

    def __len__(self):
        return len(self.image_list)

    def get_seq_len(self, seq_idx):
        return len(self.image_list[seq_idx])
    
    def get_item(self, seq_idx, img_idx):
        cam_list = self.camera_list[seq_idx]
        K, K_inv = cam_list['K'], cam_list['K_inv']

        img = Image.open(self.image_list[seq_idx][img_idx])
        if not img.mode == "RGB":
            img = img.convert("RGB")
        img = img.resize((self.image_size, self.image_size), resample=PIL.Image.Resampling.BILINEAR)
        img = np.asarray(img)
        img = (img / 127.5 - 1.0).astype(np.float32)

        ## camera
        R = cam_list['R'][img_idx]
        t = cam_list['t'][img_idx]

        Rt = np.zeros((4, 4), dtype=R.dtype)
        Rt[:3, :3] = R
        Rt[:3, 3] = t
        Rt[3, 3] = 1.0
        # camera = np.concatenate((R.reshape(-1), t.reshape(-1), K.reshape(-1), K_inv.reshape(-1)), 0)

        example = {
            'image': torch.from_numpy(img).to(torch.float32),
            'K': torch.from_numpy(K).to(torch.float32),
            'Rt': torch.from_numpy(Rt).to(torch.float32)
            # 'camera': torch.from_numpy(camera).to(torch.float32)
        }
        return example


if __name__ == "__main__":
    import logging
    dataset = RE10KDataset(task='train')
    from torch.utils.data import DataLoader
    print(len(dataset))
    dl = DataLoader(dataset, batch_size=2, shuffle=False,
                    drop_last=True, num_workers=0)
    print(len(dl))
    for batch in dl:
        for k in batch.keys():
            print(k, batch[k].shape, batch[k].max(), batch[k].min())
        import ipdb; ipdb.set_trace()
