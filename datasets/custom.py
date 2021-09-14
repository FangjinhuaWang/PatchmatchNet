from torch.utils.data import Dataset
from datasets.data_io import read_cam_file, read_pair_file, read_image
import numpy as np
import os


class MVSDataset(Dataset):
    def __init__(self, datapath, n_views=3, img_wh=(736, 416)):
        
        self.stages = 4
        self.datapath = datapath
        self.img_wh = img_wh
        
        self.metas = read_pair_file(os.path.join(self.datapath, 'pair.txt'))
        self.n_views = n_views
        
    def __len__(self):
        return len(self.metas)

    def __getitem__(self, idx):
        
        ref_view, src_views = self.metas[idx]
        # use only the reference view and first nviews-1 source views
        view_ids = [ref_view] + src_views[:self.n_views-1]

        images = []
        intrinsics = []
        extrinsics = []
        depth_min = None
        depth_max = None

        for i, vid in enumerate(view_ids):
            img_filename = os.path.join(self.datapath, f'images/{vid:08d}.jpg')
            cam_filename = os.path.join(self.datapath, f'cams/{vid:08d}_cam.txt')

            image, original_h, original_w = read_image(img_filename, max(self.img_wh))
            images.append(image.transpose([2, 0, 1]))

            intrinsic, extrinsic, depth_params = read_cam_file(cam_filename)
            intrinsic[0] *= self.img_wh[0]/original_w
            intrinsic[1] *= self.img_wh[1]/original_h
            intrinsics.append(intrinsic)
            extrinsics.append(extrinsic)

            if i == 0:  # reference view
                depth_min = depth_params[0]
                depth_max = depth_params[1]

        intrinsics = np.stack(intrinsics)
        extrinsics = np.stack(extrinsics)

        return {"images": images,               # [N][3*Hi*Wi]
                "intrinsics": intrinsics,       # N*3*3
                "extrinsics": extrinsics,       # N*4*4
                "depth_min": depth_min,         # scalar
                "depth_max": depth_max,         # scalar
                "filename": '{}/' + '{:0>8}'.format(view_ids[0]) + "{}"
                }
