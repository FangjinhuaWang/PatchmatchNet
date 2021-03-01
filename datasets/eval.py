import os
from torch.utils.data import Dataset
from datasets.data_io import *


class MVSEvalDataset(Dataset):
    def __init__(self, data_path: str, num_views: int = 10, max_dim=1024, eval_type: str = 'custom', scan_list: str = ''):

        self.metas = []
        self.stages = 4
        self.data_path = data_path
        self.max_dim = max_dim

        self.num_views = num_views

        if eval_type == 'eth3d_test':
            scans = ['botanical_garden', 'boulders', 'bridge', 'door', 'exhibition_hall', 'lecture_room', 'living_room',
                     'lounge', 'observatory', 'old_computer', 'statue', 'terrace_2']
        elif eval_type == 'eth3d_train':
            scans = ['courtyard', 'delivery_area', 'electro', 'facade', 'kicker', 'meadow', 'office', 'pipes',
                     'playground', 'relief', 'relief_2', 'terrace', 'terrains']
        elif eval_type == 'tanks_intermediate':
            scans = ['Family', 'Francis', 'Horse', 'Lighthouse', 'M60', 'Panther', 'Playground', 'Train']
        elif eval_type == 'tanks_advanced':
            scans = ['Auditorium', 'Ballroom', 'Courtroom', 'Museum', 'Palace', 'Temple']
        elif os.path.isfile(scan_list):
            with open(scan_list) as f:
                scans = [line.rstrip() for line in f.readlines()]
        else:
            scans = ['']

        self.metas = []
        for scan in scans:
            pair_data = read_pair_file(os.path.join(self.data_path, scan, 'pair.txt'))
            self.metas += [(scan, ref, src) for ref, src in pair_data]

    def __len__(self):
        return len(self.metas)

    def __getitem__(self, idx):

        scan, ref_view, src_views = self.metas[idx]
        # use only the reference view and first num_views-1 source views
        view_ids = [ref_view] + src_views[:self.num_views - 1]

        images = []
        intrinsics = []
        extrinsics = []
        depth_params = np.empty(2, dtype=np.float32)

        for i, vid in enumerate(view_ids):
            img_filename = os.path.join(self.data_path, scan, f'images/{vid:08d}.jpg')
            proj_mat_filename = os.path.join(self.data_path, scan, f'cams/{vid:08d}_cam.txt')

            image, original_h, original_w = read_image(img_filename, self.max_dim)
            images.append(image.transpose([2, 0, 1]))

            intrinsic, extrinsic, depth_params_ = read_cam_file(proj_mat_filename)

            intrinsic[0] *= image.shape[1] / original_w
            intrinsic[1] *= image.shape[0] / original_h
            intrinsics.append(intrinsic)
            extrinsics.append(extrinsic)

            if i == 0:  # reference view
                depth_params = depth_params_

        intrinsics = np.stack(intrinsics)
        extrinsics = np.stack(extrinsics)

        return {'images': images,  # List[Tensor]: [N][3,H0,W0], N is number of images
                'intrinsics': intrinsics,  # Tensor: [N,3,3]
                'extrinsics': extrinsics,  # Tensor: [N,4,4]
                'depth_params': depth_params,  # Tensor: [2]
                'filename': os.path.join(scan, '{}', '{:0>8}'.format(view_ids[0]) + '{}')
                }
