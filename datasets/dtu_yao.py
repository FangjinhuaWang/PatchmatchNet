from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image
from datasets.data_io import *
import cv2
import random


class MVSDataset(Dataset):
    def __init__(self, datapath, listfile, mode, nviews, robust_train = False):
        super(MVSDataset, self).__init__()

        self.stages = 4
        self.datapath = datapath
        self.listfile = listfile
        self.mode = mode
        self.nviews = nviews
        
        self.robust_train = robust_train
        

        assert self.mode in ["train", "val", "test"]
        self.metas = self.build_list()

    def build_list(self):
        metas = []
        with open(self.listfile) as f:
            scans = f.readlines()
            scans = [line.rstrip() for line in scans]

        for scan in scans:
            pair_file = "Cameras_1/pair.txt"
            
            with open(os.path.join(self.datapath, pair_file)) as f:
                self.num_viewpoint = int(f.readline())
                # viewpoints (49)
                for view_idx in range(self.num_viewpoint):
                    ref_view = int(f.readline().rstrip())
                    src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
                    # light conditions 0-6
                    for light_idx in range(7):
                        metas.append((scan, light_idx, ref_view, src_views))
        print("dataset", self.mode, "metas:", len(metas))
        return metas

    def __len__(self):
        return len(self.metas)

    def read_cam_file(self, filename):
        with open(filename) as f:
            lines = f.readlines()
            lines = [line.rstrip() for line in lines]
        # extrinsics: line [1,5), 4x4 matrix
        extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
        # intrinsics: line [7-10), 3x3 matrix
        intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))
        
        depth_min = float(lines[11].split()[0])
        depth_max = float(lines[11].split()[1])
        return intrinsics, extrinsics, depth_min, depth_max

    def read_img(self, filename):
        img = Image.open(filename)
        # scale 0~255 to 0~1
        
        np_img = np.array(img, dtype=np.float32) / 255.
        h, w, _ = np_img.shape
        np_img_ms = {
            "stage_3": cv2.resize(np_img, (w//8, h//8), interpolation=cv2.INTER_LINEAR), 
            "stage_2": cv2.resize(np_img, (w//4, h//4), interpolation=cv2.INTER_LINEAR),
            "stage_1": cv2.resize(np_img, (w//2, h//2), interpolation=cv2.INTER_LINEAR),
            "stage_0": np_img
        }
        return np_img_ms
        
    def read_depth(self, filename):
        return np.array(read_pfm(filename)[0], dtype=np.float32)

    def prepare_img(self, hr_img):
        # original w,h: 1600, 1200; downsample -> 800, 600 ; crop -> 640, 512
        #downsample
        h, w = hr_img.shape
        hr_img_ds = cv2.resize(hr_img, (w//2, h//2), interpolation=cv2.INTER_NEAREST)
        #crop
        h, w = hr_img_ds.shape
        target_h, target_w = 512, 640
        start_h, start_w = (h - target_h)//2, (w - target_w)//2
        hr_img_crop = hr_img_ds[start_h: start_h + target_h, start_w: start_w + target_w]

        return hr_img_crop

    def read_mask_hr(self, filename):
        img = Image.open(filename)
        np_img = np.array(img, dtype=np.float32)
        np_img = (np_img > 10).astype(np.float32)
        np_img = self.prepare_img(np_img)

        h, w = np_img.shape
        np_img_ms = {
            "stage_3": cv2.resize(np_img, (w//8, h//8), interpolation=cv2.INTER_NEAREST),
            "stage_2": cv2.resize(np_img, (w//4, h//4), interpolation=cv2.INTER_NEAREST),
            "stage_1": cv2.resize(np_img, (w//2, h//2), interpolation=cv2.INTER_NEAREST),
            "stage_0": np_img
        }
        return np_img_ms
        

    def read_depth_hr(self, filename):
        
        depth_hr = np.array(read_pfm(filename)[0], dtype=np.float32)
        depth_hr = np.squeeze(depth_hr,2)
        depth_lr = self.prepare_img(depth_hr)

        h, w = depth_lr.shape
        depth_lr_ms = {
            
            "stage_3": cv2.resize(depth_lr, (w//8, h//8), interpolation=cv2.INTER_NEAREST),
            "stage_2": cv2.resize(depth_lr, (w//4, h//4), interpolation=cv2.INTER_NEAREST),
            "stage_1": cv2.resize(depth_lr, (w//2, h//2), interpolation=cv2.INTER_NEAREST),
            "stage_0": depth_lr
        }
        return depth_lr_ms


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
            img_filename = os.path.join(self.datapath,
                                        'Rectified/{}_train/rect_{:0>3}_{}_r5000.png'.format(scan, vid + 1, light_idx))
            
            mask_filename_hr = os.path.join(self.datapath, 'Depths_raw/{}/depth_visual_{:0>4}.png'.format(scan, vid))
            depth_filename_hr = os.path.join(self.datapath, 'Depths_raw/{}/depth_map_{:0>4}.pfm'.format(scan, vid))
            proj_mat_filename = os.path.join(self.datapath, 'Cameras_1/train/{:0>8}_cam.txt').format(vid)

            imgs = self.read_img(img_filename)
            imgs_0.append(imgs['stage_0'])
            imgs_1.append(imgs['stage_1'])
            imgs_2.append(imgs['stage_2'])
            imgs_3.append(imgs['stage_3'])

            # here, the intrinsics from file is already adjusted to the downsampled size of feature 1/4H0 * 1/4W0
            intrinsics, extrinsics, depth_min_, depth_max_ = self.read_cam_file(proj_mat_filename)

            proj_mat = extrinsics.copy()
            intrinsics[:2,:] *= 0.5
            proj_mat[:3, :4] = np.matmul(intrinsics, proj_mat[:3, :4])
            proj_matrices_3.append(proj_mat)

            proj_mat = extrinsics.copy()
            intrinsics[:2,:] *= 2
            proj_mat[:3, :4] = np.matmul(intrinsics, proj_mat[:3, :4])
            proj_matrices_2.append(proj_mat)

            proj_mat = extrinsics.copy()
            intrinsics[:2,:] *= 2
            proj_mat[:3, :4] = np.matmul(intrinsics, proj_mat[:3, :4])
            proj_matrices_1.append(proj_mat)

            proj_mat = extrinsics.copy()
            intrinsics[:2,:] *= 2
            proj_mat[:3, :4] = np.matmul(intrinsics, proj_mat[:3, :4])
            proj_matrices_0.append(proj_mat)

            if i == 0:  # reference view
                depth_min = depth_min_
                depth_max = depth_max_
                
                mask = self.read_mask_hr(mask_filename_hr)
                depth = self.read_depth_hr(depth_filename_hr)
                for l in range(self.stages):
                    mask[f'stage_{l}'] = np.expand_dims(mask[f'stage_{l}'],2)
                    mask[f'stage_{l}'] = mask[f'stage_{l}'].transpose([2,0,1])
                    depth[f'stage_{l}'] = np.expand_dims(depth[f'stage_{l}'],2)
                    depth[f'stage_{l}'] = depth[f'stage_{l}'].transpose([2,0,1])
                
                

                
        # imgs: N*3*H0*W0, N is number of images
        imgs_0 = np.stack(imgs_0).transpose([0, 3, 1, 2])
        imgs_1 = np.stack(imgs_1).transpose([0, 3, 1, 2])
        imgs_2 = np.stack(imgs_2).transpose([0, 3, 1, 2])
        imgs_3 = np.stack(imgs_3).transpose([0, 3, 1, 2])
        
        imgs = {}
        imgs['stage_0'] = imgs_0
        imgs['stage_1'] = imgs_1
        imgs['stage_2'] = imgs_2
        imgs['stage_3'] = imgs_3
        
        # proj_matrices: N*4*4
        proj_matrices_0 = np.stack(proj_matrices_0)
        proj_matrices_1 = np.stack(proj_matrices_1)
        proj_matrices_2 = np.stack(proj_matrices_2)
        proj_matrices_3 = np.stack(proj_matrices_3)
        
        proj={}
        proj['stage_3']=proj_matrices_3
        proj['stage_2']=proj_matrices_2
        proj['stage_1']=proj_matrices_1
        proj['stage_0']=proj_matrices_0
        


        
        # data is numpy array
        return {"imgs": imgs,                   # N*3*H0*W0
                "proj_matrices": proj, # N*4*4
                "depth": depth,                 # 1*H0 * W0
                "depth_min": depth_min,         # scalar
                "depth_max": depth_max,         # scalar
                "mask": mask}                   # 1*H0 * W0

