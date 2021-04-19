import os
import random
from typing import Dict, List, Tuple

import numpy as np
from torch.utils.data import Dataset

from datasets.data_io import read_cam_file, read_image, read_image_dictionary, read_map, read_pair_file


class MVSDataset(Dataset):
    def __init__(self, data_path: str, num_views: int = 10, max_dim: int = -1, scan_list: str = '',
                 robust_train: bool = False, num_light_idx: int = -1, cam_folder: str = 'cams',
                 pair_path: str = 'pair.txt', image_folder: str = 'images', depth_folder: str = 'depth_gt',
                 index_path: str = None):
        super(MVSDataset, self).__init__()

        self.data_path = data_path
        self.num_views = num_views
        self.max_dim = max_dim
        self.robust_train = robust_train
        self.cam_folder = cam_folder
        self.depth_folder = depth_folder
        self.image_folder = image_folder

        if os.path.isfile(scan_list):
            with open(scan_list) as f:
                scans = [line.rstrip() for line in f.readlines()]
        else:
            scans = ['']

        if num_light_idx > 0:
            prefixes = [str(idx) for idx in range(num_light_idx)]
        else:
            prefixes = ['']

        self.metas: List[Tuple[str, str, int, List[int]]] = []
        for scan in scans:
            pair_data = read_pair_file(os.path.join(self.data_path, scan, pair_path))
            for prefix in prefixes:
                self.metas += [(scan, prefix, ref, src) for ref, src in pair_data]

        if index_path is None:
            self.has_index = False
            self.image_index: Dict[int, str] = {}
        else:
            self.has_index = True
            self.image_index = read_image_dictionary(os.path.join(self.data_path, index_path))

    def __len__(self):
        return len(self.metas)

    def __getitem__(self, idx):

        scan, prefix, ref_view, src_views = self.metas[idx]
        # use only the reference view and first num_views-1 source views
        num_src_views = min(len(src_views), self.num_views)
        if self.robust_train:
            index = random.sample(range(len(src_views)), num_src_views)
            view_ids = [ref_view] + [src_views[i] for i in index]
        else:
            view_ids = [ref_view] + src_views[:num_src_views]

        images = []
        intrinsics = []
        extrinsics = []
        depth_params = np.empty(2, dtype=np.float32)
        depth_gt = np.empty(0)
        mask = np.empty(0)

        for view_index, view_id in enumerate(view_ids):
            if self.has_index:
                img_filename = os.path.join(self.data_path, scan, self.image_folder, self.image_index[view_id])
            else:
                img_filename = os.path.join(
                    self.data_path, scan, self.image_folder, prefix, '{:0>8}.jpg'.format(view_id))

            image, original_h, original_w = read_image(img_filename, self.max_dim)
            images.append(image.transpose([2, 0, 1]))

            cam_filename = os.path.join(self.data_path, scan, self.cam_folder, '{:0>8}_cam.txt'.format(view_id))
            intrinsic, extrinsic, depth_params_ = read_cam_file(cam_filename)

            intrinsic[0] *= image.shape[1] / original_w
            intrinsic[1] *= image.shape[0] / original_h
            intrinsics.append(intrinsic)
            extrinsics.append(extrinsic)

            if view_index == 0:  # reference view
                depth_params = depth_params_
                if self.has_index:
                    depth_gt_filename = os.path.join(self.data_path, scan, self.depth_folder, self.image_index[view_id])
                    depth_gt_filename = os.path.splitext(depth_gt_filename.replace('_undistorted', ''))[0] + '.pfm'
                else:
                    depth_gt_filename = os.path.join(
                        self.data_path, scan, self.depth_folder, '{:0>8}.pfm'.format(view_id))

                if os.path.isfile(depth_gt_filename):
                    depth_gt = read_map(depth_gt_filename, self.max_dim).squeeze().copy()
                    # Create mask from GT depth map
                    mask = (depth_gt > depth_params[0]).astype(np.float32)

        intrinsics = np.stack(intrinsics)
        extrinsics = np.stack(extrinsics)

        return {'images': images,  # List[Tensor]: [N][3,H0,W0], N is number of images
                'intrinsics': intrinsics,  # Tensor: [N,3,3]
                'extrinsics': extrinsics,  # Tensor: [N,4,4]
                'depth_params': depth_params,  # Tensor: [2]
                'depth_gt': depth_gt,  # Tensor: [H0,W0] if exists
                'mask': mask,  # Tensor: [H0,W0] if exists
                'filename': os.path.join(scan, '{}', '{:0>8}'.format(view_ids[0]) + '{}')
                }
