from torch.utils.data import Dataset
from datasets.data_io import read_cam_file, read_pair_file, read_image, read_map
from typing import List, Tuple
from PIL import Image
import numpy as np
import os
import cv2
import random


def prepare_img(hr_img):
    # original w,h: 1600, 1200; downsample -> 800, 600 ; crop -> 640, 512
    # downsample
    h, w = hr_img.shape
    hr_img_ds = cv2.resize(hr_img, (w//2, h//2), interpolation=cv2.INTER_NEAREST)
    # crop
    h, w = hr_img_ds.shape
    target_h, target_w = 512, 640
    start_h, start_w = (h - target_h)//2, (w - target_w)//2
    hr_img_crop = hr_img_ds[start_h: start_h + target_h, start_w: start_w + target_w]

    return hr_img_crop


def read_depth_mask(stages, filename, mask_filename, depth_min, depth_max):
    depth = np.array(read_map(filename), dtype=np.float32)
    depth = prepare_img(np.squeeze(depth, 2))

    mask = np.array(Image.open(mask_filename), dtype=np.float32)
    mask = (mask > 10).astype(np.float32)
    mask = prepare_img(mask).astype(bool)
    mask = mask & (depth >= depth_min) & (depth <= depth_max)
    mask = mask.astype(np.float32)

    h, w = depth.shape
    depth_ms = {}
    mask_ms = {}

    for i in range(stages):
        depth_cur = cv2.resize(depth, (w//(2**i), h//(2**i)), interpolation=cv2.INTER_NEAREST)
        mask_cur = cv2.resize(mask, (w//(2**i), h//(2**i)), interpolation=cv2.INTER_NEAREST)
        depth_ms[f"stage_{i}"] = depth_cur
        mask_ms[f"stage_{i}"] = mask_cur

    return depth_ms, mask_ms


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

        imgs_0 = []
        imgs_1 = []
        imgs_2 = []
        imgs_3 = []

        mask = None
        depth = None
        depth_min = None
        depth_max = None
        
        proj_matrices_0 = []
        proj_matrices_1 = []
        proj_matrices_2 = []
        proj_matrices_3 = []

        for i, vid in enumerate(view_ids):
            # NOTE that the id in image file names is from 1 to 49 (not 0~48)
            img_filename = os.path.join(
                self.datapath, 'Rectified/{}_train/rect_{:0>3}_{}_r5000.png'.format(scan, vid + 1, light_idx))
            mask_filename_hr = os.path.join(self.datapath, 'Depths_raw/{}/depth_visual_{:0>4}.png'.format(scan, vid))
            depth_filename_hr = os.path.join(self.datapath, 'Depths_raw/{}/depth_map_{:0>4}.pfm'.format(scan, vid))
            proj_mat_filename = os.path.join(self.datapath, 'Cameras_1/train/{:0>8}_cam.txt').format(vid)

            image, original_h, original_w = read_image(img_filename)
            imgs_0.append(image)
            imgs_1.append(cv2.resize(image, (original_w//2, original_h//2), interpolation=cv2.INTER_LINEAR))
            imgs_2.append(cv2.resize(image, (original_w//4, original_h//4), interpolation=cv2.INTER_LINEAR))
            imgs_3.append(cv2.resize(image, (original_w//8, original_h//8), interpolation=cv2.INTER_LINEAR))

            # here, the intrinsics from file is already adjusted to the downsampled size of feature 1/4H0 * 1/4W0
            intrinsics, extrinsics, depth_params = read_cam_file(proj_mat_filename)

            proj_mat = extrinsics.copy()
            intrinsics[:2, :] *= 0.5
            proj_mat[:3, :4] = np.matmul(intrinsics, proj_mat[:3, :4])
            proj_matrices_3.append(proj_mat)

            proj_mat = extrinsics.copy()
            intrinsics[:2, :] *= 2
            proj_mat[:3, :4] = np.matmul(intrinsics, proj_mat[:3, :4])
            proj_matrices_2.append(proj_mat)

            proj_mat = extrinsics.copy()
            intrinsics[:2, :] *= 2
            proj_mat[:3, :4] = np.matmul(intrinsics, proj_mat[:3, :4])
            proj_matrices_1.append(proj_mat)

            proj_mat = extrinsics.copy()
            intrinsics[:2, :] *= 2
            proj_mat[:3, :4] = np.matmul(intrinsics, proj_mat[:3, :4])
            proj_matrices_0.append(proj_mat)

            if i == 0:  # reference view
                depth_min = depth_params[0]
                depth_max = depth_params[1]
                depth, mask = read_depth_mask(self.stages, depth_filename_hr, mask_filename_hr, depth_min, depth_max)
                for j in range(self.stages):
                    mask[f'stage_{j}'] = np.expand_dims(mask[f'stage_{j}'], 2)
                    mask[f'stage_{j}'] = mask[f'stage_{j}'].transpose([2, 0, 1])
                    depth[f'stage_{j}'] = np.expand_dims(depth[f'stage_{j}'], 2)
                    depth[f'stage_{j}'] = depth[f'stage_{j}'].transpose([2, 0, 1])

        # imgs: N*3*H0*W0, N is number of images
        imgs = {
            'stage_0': np.stack(imgs_0).transpose([0, 3, 1, 2]),
            'stage_1': np.stack(imgs_1).transpose([0, 3, 1, 2]),
            'stage_2': np.stack(imgs_2).transpose([0, 3, 1, 2]),
            'stage_3': np.stack(imgs_3).transpose([0, 3, 1, 2])
        }
        # proj_matrices: N*4*4
        proj = {
            'stage_3': np.stack(proj_matrices_3),
            'stage_2': np.stack(proj_matrices_2),
            'stage_1': np.stack(proj_matrices_1),
            'stage_0': np.stack(proj_matrices_0)
        }

        # data is numpy array
        return {"imgs": imgs,                   # N*3*H0*W0
                "proj_matrices": proj,          # N*4*4
                "depth": depth,                 # 1*H0*W0
                "depth_min": depth_min,         # scalar
                "depth_max": depth_max,         # scalar
                "mask": mask}                   # 1*H0*W0
