from torch.utils.data import Dataset
from datasets.data_io import read_cam_file, read_pair_file, read_image
from typing import List, Tuple
import numpy as np
import os


class MVSDataset(Dataset):
    def __init__(self, datapath, split='intermediate', n_views=3, img_wh=(1920, 1056)):
        
        self.stages = 4
        self.datapath = datapath
        self.img_wh = img_wh
        self.split = split
        self.n_views = n_views

        scans = []
        if self.split == 'intermediate':
            scans = ['Family', 'Francis', 'Horse', 'Lighthouse', 'M60', 'Panther', 'Playground', 'Train']
            self.image_sizes = {'Family': (1920, 1080),
                                'Francis': (1920, 1080),
                                'Horse': (1920, 1080),
                                'Lighthouse': (2048, 1080),
                                'M60': (2048, 1080),
                                'Panther': (2048, 1080),
                                'Playground': (1920, 1080),
                                'Train': (1920, 1080)}
        elif self.split == 'advanced':
            scans = ['Auditorium', 'Ballroom', 'Courtroom', 'Museum', 'Palace', 'Temple']
            self.image_sizes = {'Auditorium': (1920, 1080),
                                'Ballroom': (1920, 1080),
                                'Courtroom': (1920, 1080),
                                'Museum': (1920, 1080),
                                'Palace': (1920, 1080),
                                'Temple': (1920, 1080)}

        self.metas: List[Tuple[str, int, List[int]]] = []
        for scan in scans:
            pair_data = read_pair_file(os.path.join(self.datapath, self.split, scan, 'pair.txt'))
            self.metas += [(scan, ref, src) for ref, src in pair_data]

    def __len__(self):
        return len(self.metas)

    def __getitem__(self, idx):
        
        scan, ref_view, src_views = self.metas[idx]
        # use only the reference view and first nviews-1 source views
        view_ids = [ref_view] + src_views[:self.n_views-1]
        img_w, img_h = self.image_sizes[scan]

        images = []
        intrinsics = []
        extrinsics = []
        depth_min = None
        depth_max = None
        
        for i, vid in enumerate(view_ids):
            img_filename = os.path.join(self.datapath, self.split, scan, f'images/{vid:08d}.jpg')
            proj_mat_filename = os.path.join(self.datapath, self.split, scan, f'cams_1/{vid:08d}_cam.txt')
            
            image, _, _ = read_image(img_filename, max(self.img_wh))
            images.append(image.transpose([2, 0, 1]))

            intrinsic, extrinsic, depth_params = read_cam_file(proj_mat_filename)
            intrinsic[0] *= self.img_wh[0]/img_w
            intrinsic[1] *= self.img_wh[1]/img_h
            intrinsics.append(intrinsic)
            extrinsics.append(extrinsic)

            if i == 0:  # reference view
                depth_min = depth_params[0]
                depth_max = depth_params[1]

        intrinsics = np.stack(intrinsics)
        extrinsics = np.stack(extrinsics)

        return {"images": images,               # [N][3*H0*W0]
                "intrinsics": intrinsics,       # N*3*3
                "extrinsics": extrinsics,       # N*4*4
                "depth_min": depth_min,         # scalar
                "depth_max": depth_max,         # scalar
                "filename": scan + '/{}/' + '{:0>8}'.format(view_ids[0]) + "{}"
                }  
