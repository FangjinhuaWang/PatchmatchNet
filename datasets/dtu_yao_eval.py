from torch.utils.data import Dataset
from datasets.data_io import read_cam_file, read_pair_file, read_image
from typing import List, Tuple
import numpy as np
import os


class MVSDataset(Dataset):
    def __init__(self, datapath, listfile, mode, nviews, img_wh=(1600, 1200)):
        super(MVSDataset, self).__init__()
        
        self.stages = 4
        self.datapath = datapath
        self.nviews = nviews
        self.img_wh = img_wh

        assert mode == "test"

        with open(listfile) as f:
            scans = f.readlines()
            scans = [line.rstrip() for line in scans]

        self.metas: List[Tuple[str, int, List[int]]] = []
        for scan in scans:
            pair_data = read_pair_file(os.path.join(self.datapath, scan, 'pair.txt'))
            self.metas += [(scan, ref, src) for ref, src in pair_data]
        print("dataset", mode, "metas:", len(self.metas))

    def __len__(self):
        return len(self.metas)

    def __getitem__(self, idx):
        meta = self.metas[idx]
        scan, ref_view, src_views = meta
        # use only the reference view and first nviews-1 source views
        view_ids = [ref_view] + src_views[:self.nviews - 1]
        img_w = 1600
        img_h = 1200

        images = []
        intrinsics = []
        extrinsics = []
        depth_min = None
        depth_max = None

        for i, vid in enumerate(view_ids):
            img_filename = os.path.join(self.datapath, '{}/images/{:0>8}.jpg'.format(scan, vid))
            cam_filename = os.path.join(self.datapath, '{}/cams_1/{:0>8}_cam.txt'.format(scan, vid))

            image, original_h, original_w = read_image(img_filename, max(self.img_wh))
            images.append(image.transpose([2, 0, 1]))

            intrinsic, extrinsic, depth_params = read_cam_file(cam_filename)
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
                "filename": scan + '/{}/' + '{:0>8}'.format(view_ids[0]) + "{}"}
