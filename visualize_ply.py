from argparse import ArgumentParser
import open3d as o3d
import numpy as np
import os

# from https://github.com/intel-isl/Open3D/blob/master/examples/Python/Advanced/load_save_viewpoint.py
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--logdir', type=str, required=True,
                        help='the scan to visualize')
    parser.add_argument('--scan', type=str, required=True,
                        help='the scan to visualize')
    parser.add_argument('--dataset', type=str, default='dtu',
                        help='identify dataset')
    parser.add_argument('--use_viewpoint', default=False, action='store_true',
                        help='use precalculated viewpoint')
    parser.add_argument('--save_viewpoint', default=False, action='store_true',
                        help='save this viewpoint')

    args = parser.parse_args()
    if args.dataset == 'dtu':
        path =  os.path.join(args.logdir, 'patchmatchnet{:0>3}_l3.ply'.format(args.scan))
    else:
        path =  os.path.join(args.logdir, args.scan+'.ply')
    pcd = o3d.io.read_point_cloud(path)
    print(f'{args.scan} contains {len(pcd.points)/1e6:.2f} M points')
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    ctr = vis.get_view_control()
    opt = vis.get_render_option()
    opt.point_size = 1.0
    opt.background_color = np.array([1,1, 1.0])
    vis.add_geometry(pcd)
    
    if args.use_viewpoint:
        # param = o3d.io.read_pinhole_camera_parameters(f"viewpoints/{args.dataset}/viewpoint.json")
        param = o3d.io.read_pinhole_camera_parameters("viewpoints/dtu/viewpoint.json")
        ctr.convert_from_pinhole_camera_parameters(param)
        vis.run()
    elif args.save_viewpoint:
        vis.run()
        param = ctr.convert_to_pinhole_camera_parameters()
        # o3d.io.write_pinhole_camera_parameters(f"viewpoints/{args.dataset}/viewpoint.json", param)
        o3d.io.write_pinhole_camera_parameters("viewpoints/dtu/viewpoint.json", param)
    else:
        vis.run()
    
    vis.run()
    vis.destroy_window()
