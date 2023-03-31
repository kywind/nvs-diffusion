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
                       size=256, 
                       flip_p=0.5,
                       interval=1,
                       low=3,
                       high=20,
                       clip_length=2,
                       n_videos=0,
                       eval_interval=1):
        self.task = task
        if self.task == 'train':
            self.data_path = '/data/yangfu2/data/re10k/train/'
            self.camera_dir = '/data/yangfu2/data/re10k/anns/train/'
            self.flip_p = flip_p
        elif self.task == 'val':
            self.data_path = '/data/yangfu2/data/re10k/sample/'
            self.camera_dir = '/data/yangfu2/data/re10k/anns/sample/'
            self.flip_p = 0
        else:
            assert self.task == 'test'
            self.data_path = '/data/yangfu2/data/re10k/test/'
            self.camera_dir = '/data/yangfu2/data/re10k/anns/test/'
            self.flip_p = 0

        self.image_size = size
        self.clip_length = clip_length
        self.low = low
        self.high = high
        self.interval = interval
        self.eval_interval = eval_interval

        self.IMG_EXT = ['png', 'jpg']
        if self.data_path is None:
            raise RuntimeError(f"{task} dataset not exist.")
        self.video_paths = sorted(glob.glob(self.data_path + '/*/'))
        if n_videos != 0:
            self.video_paths = self.video_paths[:n_videos]

        print('loading data...')
        self.image_list = []
        self.camera_list = []
        self.idx_list = []
        self.length = 0
        self.video_count = 0
        for idx in tqdm(range(len(self.video_paths))):
            vpath = self.video_paths[idx]
            frames = sum([sorted(glob.glob(vpath + f'/*.{ext}')) for ext in self.IMG_EXT],[])[::self.interval]
            if len(frames) == 0: continue
            self.image_list.append(frames)
            self.camera_list.append(self.read_camera(frames))
            self.length += len(frames)

            self.idx_list.extend(list(zip([self.video_count] * len(frames), list(range(len(frames))))))
            self.video_count += 1
    
        print(f'{self.task} dataset length:', self.length)
        self.flip = transforms.RandomHorizontalFlip(p=self.flip_p)

    
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
            # rescale = self.image_size
            # intr = intr * rescale
            # norm_factor = self.image_size / 2
            # f_ndc = np.array([intr[0, 0] / norm_factor, intr[1, 1]/ norm_factor])
            # pp_ndc = np.array(
            #     [(intr[0, 2] - self.image_size/2) / norm_factor, 
            #      (intr[1, 2] - self.image_size/2) / norm_factor])
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
                cam_dict['K_inv'] = np.linalg.inv(intr)
            else:
                assert abs((cam_dict['K'] - intr).sum()) < 1e-5

            extr = extr.reshape(3, 4)
            R = extr[:, :3].astype(np.float32)  # 3,3
            t = extr[:, 3].astype(np.float32)  # 3,

            cam_dict['R'].append(R)
            cam_dict['t'].append(t)
            
        return cam_dict

    def __len__(self):
        return self.length

    def __getitem__(self, index):

        seq_idx, img_idx = self.idx_list[index]

        cam_list = self.camera_list[seq_idx]
        K, K_inv = cam_list['K'], cam_list['K_inv']

        clip = []
        for idx in range(self.clip_length):
            if idx == 0:
                pass
            else:
                if self.task == 'train':
                    sample_list = [i for i in range(len(self.image_list[seq_idx])) 
                                    if abs(i-img_idx) >= self.low and abs(i-img_idx) <= self.high]
                    if len(sample_list) == 0:
                        sample_list = list(range(len(self.image_list[seq_idx])))
                    img_idx = random.sample(sample_list, 1)[0]
                else:
                    if img_idx + self.eval_interval < len(self.image_list[seq_idx]):
                        img_idx = img_idx + self.eval_interval

            # example = dict((k, self.labels[k][index]) for k in self.labels)
            # image = Image.open(imagefn)

            img = Image.open(self.image_list[seq_idx][img_idx])
            if not img.mode == "RGB":
                img = img.convert("RGB")

            ## default to score-sde preprocessing
            # img = np.array(image).astype(np.uint8)
            # crop = min(img.shape[0], img.shape[1])
            # h, w = img.shape[0], img.shape[1]
            # img = img[(h - crop) // 2:(h + crop) // 2,
            #       (w - crop) // 2:(w + crop) // 2]

            # img = Image.fromarray(img)
            img = img.resize((self.image_size, self.image_size), resample=PIL.Image.Resampling.BILINEAR)
            # assert img.shape[0] == img.shape[1]

            # img = self.flip(img)
            img = np.asarray(img).astype(np.uint8)
            img = (img / 127.5 - 1.0).astype(np.float32)

            ## camera
            R = cam_list['R'][img_idx]
            t = cam_list['t'][img_idx]

            if idx == 0: 
                R0 = R
                R0_inv = R0.transpose(-1,-2)
                t0 = t

            R_rel = R @ R0_inv
            t_rel = t - R_rel @ t0

            clip.append((img, R, t, R_rel, t_rel))
        
            # post-process using size
            # if self.image_size is not None and (self.image_size[0] != h or self.image_size[1] != w):
            #     K[0,:] = K[0,:] * self.image_size[1] / w
            #     K[1,:] = K[1,:] * self.image_size[0] / h

        imgs, Rs, ts, R_rels, t_rels = (np.stack(item, 0) for item in zip(*clip))
        image = imgs[1]
        ref_image = imgs[0]
        camera = np.concatenate((R_rels[1].reshape(-1), t_rels[1].reshape(-1), K.reshape(-1), K_inv.reshape(-1)), 0)
        example = {
            'image': torch.from_numpy(image).to(torch.float32),
            'ref_image': torch.from_numpy(ref_image).to(torch.float32),
            'camera': torch.from_numpy(camera).to(torch.float32),
            'index': torch.tensor([seq_idx, img_idx]).to(torch.float32)
        }
        return example


class RE10KTrain(RE10KDataset):
    def __init__(self, **kwargs):
        super().__init__(task='train', **kwargs)


class RE10KValidation(RE10KDataset):
    def __init__(self, flip_p=0., **kwargs):
        super().__init__(task='val', flip_p=flip_p, **kwargs)


class RE10KTest(RE10KDataset):
    def __init__(self, flip_p=0., **kwargs):
        super().__init__(task='test', flip_p=flip_p, **kwargs)



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
