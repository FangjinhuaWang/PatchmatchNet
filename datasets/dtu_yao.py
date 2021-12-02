from torch.utils.data import Dataset
from datasets.data_io import read_cam_file, read_pair_file, read_image, read_map
from typing import List, Tuple
from PIL import Image
import cv2
import numpy as np
import os
import random


def prepare_img(hr_img: np.ndarray) -> np.ndarray:
    # original w,h: 1600, 1200; downsample -> 800, 600 ; crop -> 640, 512
    # downsample
    h, w = hr_img.shape
    hr_img_ds = cv2.resize(hr_img, (w // 2, h // 2), interpolation=cv2.INTER_NEAREST)
    # crop
    h, w = hr_img_ds.shape
    target_h, target_w = 512, 640
    start_h, start_w = (h - target_h) // 2, (w - target_w) // 2
    hr_img_crop = hr_img_ds[start_h: start_h + target_h, start_w: start_w + target_w]

    return np.expand_dims(hr_img_crop, 2).transpose([2, 0, 1])


def read_mask_hr(filename: str) -> np.ndarray:
    return prepare_img((np.array(Image.open(filename), dtype=np.float32) > 10).astype(np.float32))


def read_depth_hr(filename: str) -> np.ndarray:
    return prepare_img(read_map(filename).squeeze(2))


class MVSDataset(Dataset):
    def __init__(self, datapath, listfile, mode, nviews, robust_train=False):
        super(MVSDataset, self).__init__()

        self.stages = 4
        self.datapath = datapath
        self.nviews = nviews
        self.robust_train = robust_train

        assert mode in ["train", "val", "test"]

        with open(listfile) as f:
            scans = f.readlines()
            scans = [line.rstrip() for line in scans]

        self.metas: List[Tuple[str, int, int, List[int]]] = []
        for scan in scans:
            pair_data = read_pair_file(os.path.join(self.datapath, "Cameras_1/pair.txt"))
            for light_idx in range(7):
                self.metas += [(scan, light_idx, ref, src) for ref, src in pair_data]
        print("dataset", mode, "metas:", len(self.metas))

    def __len__(self):
        return len(self.metas)

    def __getitem__(self, idx):
        meta = self.metas[idx]
        scan, light_idx, ref_view, src_views = meta

        # robust training strategy
        if self.robust_train:
            num_src_views = len(src_views)
            index = random.sample(range(num_src_views), self.nviews - 1)
            view_ids = [ref_view] + [src_views[i] for i in index]
        else:
            view_ids = [ref_view] + src_views[:self.nviews - 1]

        images = []
        intrinsics = []
        extrinsics = []
        mask = None
        depth = None
        depth_min = None
        depth_max = None

        for i, vid in enumerate(view_ids):
            # NOTE that the id in image file names is from 1 to 49 (not 0~48)
            img_filename = os.path.join(
                self.datapath, 'Rectified/{}_train/rect_{:0>3}_{}_r5000.png'.format(scan, vid + 1, light_idx))
            mask_filename_hr = os.path.join(self.datapath, 'Depths_raw/{}/depth_visual_{:0>4}.png'.format(scan, vid))
            depth_filename_hr = os.path.join(self.datapath, 'Depths_raw/{}/depth_map_{:0>4}.pfm'.format(scan, vid))
            cam_filename = os.path.join(self.datapath, 'Cameras_1/train/{:0>8}_cam.txt').format(vid)

            image, original_h, original_w = read_image(img_filename)
            images.append(image.transpose([2, 0, 1]))

            # here, the intrinsics from file is already adjusted to the downsampled size of feature 1/4H0 * 1/4W0
            intrinsic, extrinsic, depth_params = read_cam_file(cam_filename)
            # Need to upscale the intrinsics since the given cam file corresponds to smaller image size
            intrinsic[:2, :] *= 4.0
            intrinsics.append(intrinsic)
            extrinsics.append(extrinsic)

            if i == 0:  # reference view
                depth_min = depth_params[0]
                depth_max = depth_params[1]
                mask = read_mask_hr(mask_filename_hr)
                depth = read_depth_hr(depth_filename_hr)

        intrinsics = np.stack(intrinsics)
        extrinsics = np.stack(extrinsics)

        # data is numpy array
        return {"images": images,  # [N][3*H0*W0]
                "intrinsics": intrinsics,  # N*3*3
                "extrinsics": extrinsics,  # N*4*4
                "depth": depth,  # 1*H0*W0
                "depth_min": depth_min,  # scalar
                "depth_max": depth_max,  # scalar
                "mask": mask}  # 1*H0*W0
