import argparse
import os
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import time
from datasets import find_dataset_def
from models import *
from utils import *
import sys
from datasets.data_io import read_cam_file, read_pair_file, read_image, read_map, save_image, save_map
import cv2
from plyfile import PlyData, PlyElement

cudnn.benchmark = True

parser = argparse.ArgumentParser(description='Predict depth, filter, and fuse')
parser.add_argument('--model', default='PatchmatchNet', help='select model')

parser.add_argument('--dataset', default='tanks', help='select dataset')
parser.add_argument('--testpath', help='testing data path')
parser.add_argument('--testlist', help='testing scan list')
parser.add_argument('--split', default='intermediate', help='select data')

parser.add_argument('--batch_size', type=int, default=1, help='testing batch size')
parser.add_argument('--n_views', type=int, default=5, help='num of view')

parser.add_argument('--loadckpt', default=None, help='load a specific checkpoint')
parser.add_argument('--outdir', default='./outputs', help='output dir')
parser.add_argument('--display', action='store_true', help='display depth images and masks')

parser.add_argument('--patchmatch_iteration', nargs='+', type=int, default=[1, 2, 2],
                    help='num of iteration of patchmatch on stages 1,2,3')
parser.add_argument('--patchmatch_num_sample', nargs='+', type=int, default=[8, 8, 16],
                    help='num of generated samples in local perturbation on stages 1,2,3')
parser.add_argument('--patchmatch_interval_scale', nargs='+', type=float, default=[0.005, 0.0125, 0.025], 
                    help='normalized interval in inverse depth range to generate samples in local perturbation')
parser.add_argument('--patchmatch_range', nargs='+', type=int, default=[6, 4, 2],
                    help='fixed offset of sampling points for propogation of patchmatch on stages 1,2,3')
parser.add_argument('--propagate_neighbors', nargs='+', type=int, default=[0, 8, 16],
                    help='num of neighbors for adaptive propagation on stages 1,2,3')
parser.add_argument('--evaluate_neighbors', nargs='+', type=int, default=[9, 9, 9],
                    help='num of neighbors for adaptive matching cost aggregation of adaptive evaluation on stages 1,2,3')

parser.add_argument('--geo_pixel_thres', type=float, default=1,
                    help='pixel threshold for geometric consistency filtering')
parser.add_argument('--geo_depth_thres', type=float, default=0.01,
                    help='depth threshold for geometric consistency filtering')


# parse arguments and check
args = parser.parse_args()
print("argv:", sys.argv[1:])
print_args(args)


# run MVS model to save depth maps
def save_depth():
    # dataset, dataloader
    mvs_dataset = find_dataset_def(args.dataset)
    test_dataset = mvs_dataset(args.testpath, args.split, args.n_views)
    image_loader = DataLoader(test_dataset, args.batch_size, shuffle=False, num_workers=4, drop_last=False)

    # model
    model = PatchmatchNet(
        patchmatch_interval_scale=args.patchmatch_interval_scale,
        propagation_range=args.patchmatch_range,
        patchmatch_iteration=args.patchmatch_iteration,
        patchmatch_num_sample=args.patchmatch_num_sample,
        propagate_neighbors=args.propagate_neighbors,
        evaluate_neighbors=args.evaluate_neighbors
    )
    model = nn.DataParallel(model)
    model.cuda()

    print("loading model {}".format(args.loadckpt))
    state_dict = torch.load(args.loadckpt)
    model.load_state_dict(state_dict['model'], strict=False)
    model.eval()
    
    with torch.no_grad():
        for batch_idx, sample in enumerate(image_loader):
            start_time = time.time()
            sample_cuda = tocuda(sample)
            refined_depth, confidence, _ = model(sample_cuda["imgs"], sample_cuda["proj_matrices"],
                                                 sample_cuda["depth_min"], sample_cuda["depth_max"])
            refined_depth = tensor2numpy(refined_depth)
            confidence = tensor2numpy(confidence)

            del sample_cuda
            print('Iter {}/{}, time = {:.3f}'.format(batch_idx, len(image_loader), time.time() - start_time))
            filenames = sample["filename"]

            # save depth maps and confidence maps
            for filename, depth_est, photometric_confidence in zip(filenames, refined_depth, confidence):
                depth_filename = os.path.join(args.outdir, filename.format('depth_est', '.pfm'))
                confidence_filename = os.path.join(args.outdir, filename.format('confidence', '.pfm'))
                os.makedirs(depth_filename.rsplit('/', 1)[0], exist_ok=True)
                os.makedirs(confidence_filename.rsplit('/', 1)[0], exist_ok=True)
                # save depth maps
                depth_est = np.squeeze(depth_est, 0)
                save_map(depth_filename, depth_est)
                # save confidence maps
                save_map(confidence_filename, photometric_confidence)


# project the reference point cloud into the source view, then project back
def reproject_with_depth(depth_ref, intrinsics_ref, extrinsics_ref, depth_src, intrinsics_src, extrinsics_src):
    width, height = depth_ref.shape[1], depth_ref.shape[0]
    # step1. project reference pixels to the source view
    # reference view x, y
    x_ref, y_ref = np.meshgrid(np.arange(0, width), np.arange(0, height))
    x_ref, y_ref = x_ref.reshape([-1]), y_ref.reshape([-1])
    # reference 3D space
    xyz_ref = np.matmul(np.linalg.inv(intrinsics_ref),
                        np.vstack((x_ref, y_ref, np.ones_like(x_ref))) * depth_ref.reshape([-1]))
    # source 3D space
    xyz_src = np.matmul(np.matmul(extrinsics_src, np.linalg.inv(extrinsics_ref)),
                        np.vstack((xyz_ref, np.ones_like(x_ref))))[:3]
    # source view x, y
    k_xyz_src = np.matmul(intrinsics_src, xyz_src)
    xy_src = k_xyz_src[:2] / k_xyz_src[2:3]

    # step2. reproject the source view points with source view depth estimation
    # find the depth estimation of the source view
    x_src = xy_src[0].reshape([height, width]).astype(np.float32)
    y_src = xy_src[1].reshape([height, width]).astype(np.float32)
    sampled_depth_src = cv2.remap(depth_src, x_src, y_src, interpolation=cv2.INTER_LINEAR)
    # mask = sampled_depth_src > 0

    # source 3D space
    # NOTE that we should use sampled source-view depth_here to project back
    xyz_src = np.matmul(np.linalg.inv(intrinsics_src),
                        np.vstack((xy_src, np.ones_like(x_ref))) * sampled_depth_src.reshape([-1]))
    # reference 3D space
    xyz_reprojected = np.matmul(np.matmul(extrinsics_ref, np.linalg.inv(extrinsics_src)),
                                np.vstack((xyz_src, np.ones_like(x_ref))))[:3]
    # source view x, y, depth
    depth_reprojected = xyz_reprojected[2].reshape([height, width]).astype(np.float32)
    k_xyz_reprojected = np.matmul(intrinsics_ref, xyz_reprojected)
    xy_reprojected = k_xyz_reprojected[:2] / k_xyz_reprojected[2:3]
    x_reprojected = xy_reprojected[0].reshape([height, width]).astype(np.float32)
    y_reprojected = xy_reprojected[1].reshape([height, width]).astype(np.float32)

    return depth_reprojected, x_reprojected, y_reprojected, x_src, y_src


def check_geometric_consistency(depth_ref, intrinsics_ref, extrinsics_ref, depth_src, intrinsics_src, extrinsics_src,
                                geo_pixel_thres=1, geo_depth_thres=0.01, dynamic_consistency_check=False):
    width, height = depth_ref.shape[1], depth_ref.shape[0]
    x_ref, y_ref = np.meshgrid(np.arange(0, width), np.arange(0, height))
    depth_reprojected, x2d_reprojected, y2d_reprojected, x2d_src, y2d_src = reproject_with_depth(
        depth_ref, intrinsics_ref, extrinsics_ref, depth_src, intrinsics_src, extrinsics_src)
    # print(depth_ref.shape)
    # print(depth_reprojected.shape)
    # check |p_reproj-p_1| < 1
    dist = np.sqrt((x2d_reprojected - x_ref) ** 2 + (y2d_reprojected - y_ref) ** 2)

    # check |d_reproj-d_1| / d_1 < 0.01
    # depth_ref = np.squeeze(depth_ref, 2)
    depth_diff = np.abs(depth_reprojected - depth_ref)
    relative_depth_diff = depth_diff / depth_ref
    if dynamic_consistency_check:
        return dist, relative_depth_diff, depth_reprojected
    
    mask = np.logical_and(dist < geo_pixel_thres, relative_depth_diff < geo_depth_thres)
    
    depth_reprojected[~mask] = 0
    
    return mask, depth_reprojected, x2d_src, y2d_src


def filter_depth(
        scan_folder, out_folder, plyfilename, geo_pixel_thres, geo_depth_thres, photo_thres, img_wh, image_sizes,
        geo_mask_thres):
    # the pair file
    pair_file = os.path.join(scan_folder, "pair.txt")
    # for the final point cloud
    vertexs = []
    vertex_colors = []

    pair_data = read_pair_file(pair_file)
    original_w, original_h = image_sizes
    
    # for each reference view and the corresponding source views
    for ref_view, src_views in pair_data:
        # load the camera parameters
        ref_intrinsics, ref_extrinsics, _ = read_cam_file(
            os.path.join(scan_folder, 'cams_1/{:0>8}_cam.txt'.format(ref_view)))
        ref_intrinsics[0] *= img_wh[0]/original_w
        ref_intrinsics[1] *= img_wh[1]/original_h
        # load the reference image
        ref_img, _, _ = read_image(os.path.join(scan_folder, 'images/{:0>8}.jpg'.format(ref_view)), max(img_wh))
        
        # load the estimated depth of the reference view
        ref_depth_est = read_map(os.path.join(out_folder, 'depth_est/{:0>8}.pfm'.format(ref_view)))
        ref_depth_est = np.squeeze(ref_depth_est, 2)
        # load the photometric mask of the reference view
        confidence = read_map(os.path.join(out_folder, 'confidence/{:0>8}.pfm'.format(ref_view)))
        
        photo_mask = confidence > photo_thres
        photo_mask = np.squeeze(photo_mask, 2)

        all_srcview_depth_ests = []
        # compute the geometric mask
        geo_mask_sum = 0
        n_views = len(src_views)+1
        for i in range(n_views-1):
            src_view = src_views[i]
            
            src_intrinsics, src_extrinsics, _ = read_cam_file(
                os.path.join(scan_folder, 'cams_1/{:0>8}_cam.txt'.format(src_view)))
            src_intrinsics[0] *= img_wh[0]/original_w
            src_intrinsics[1] *= img_wh[1]/original_h

            # the estimated depth of the source view
            src_depth_est = read_map(os.path.join(out_folder, 'depth_est/{:0>8}.pfm'.format(src_view)))

            geo_mask, depth_reprojected, _, _ = check_geometric_consistency(
                ref_depth_est, ref_intrinsics, ref_extrinsics, src_depth_est, src_intrinsics, src_extrinsics,
                geo_pixel_thres, geo_depth_thres)
            geo_mask_sum += geo_mask.astype(np.int32)
            all_srcview_depth_ests.append(depth_reprojected)

        depth_est_averaged = (sum(all_srcview_depth_ests) + ref_depth_est) / (geo_mask_sum + 1)
        geo_mask = geo_mask_sum >= geo_mask_thres
        final_mask = np.logical_and(photo_mask, geo_mask)

        os.makedirs(os.path.join(out_folder, "mask"), exist_ok=True)
        save_image(os.path.join(out_folder, "mask/{:0>8}_photo.png".format(ref_view)), photo_mask)
        save_image(os.path.join(out_folder, "mask/{:0>8}_geo.png".format(ref_view)), geo_mask)
        save_image(os.path.join(out_folder, "mask/{:0>8}_final.png".format(ref_view)), final_mask)

        print("processing {}, ref-view{:0>2}, geo_mask:{:3f} photo_mask:{:3f} final_mask: {:3f}".format(
            scan_folder, ref_view, geo_mask.mean(), photo_mask.mean(), final_mask.mean()))

        if args.display:
            cv2.imshow('ref_img', ref_img[:, :, ::-1])
            cv2.imshow('ref_depth', ref_depth_est)
            cv2.imshow('ref_depth * photo_mask', ref_depth_est * photo_mask.astype(np.float32))
            cv2.imshow('ref_depth * geo_mask', ref_depth_est * geo_mask.astype(np.float32))
            cv2.imshow('ref_depth * mask', ref_depth_est * final_mask.astype(np.float32))
            cv2.waitKey(1)

        height, width = depth_est_averaged.shape[:2]
        x, y = np.meshgrid(np.arange(0, width), np.arange(0, height))
        
        valid_points = final_mask
        # print("valid_points", valid_points.mean())
        x, y, depth = x[valid_points], y[valid_points], depth_est_averaged[valid_points]
        
        color = ref_img[valid_points]
        xyz_ref = np.matmul(np.linalg.inv(ref_intrinsics), np.vstack((x, y, np.ones_like(x))) * depth)
        xyz_world = np.matmul(np.linalg.inv(ref_extrinsics), np.vstack((xyz_ref, np.ones_like(x))))[:3]
        vertexs.append(xyz_world.transpose((1, 0)))
        vertex_colors.append((color * 255).astype(np.uint8))

    vertexs = np.concatenate(vertexs, axis=0)
    vertex_colors = np.concatenate(vertex_colors, axis=0)
    vertexs = np.array([tuple(v) for v in vertexs], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    vertex_colors = np.array([tuple(v) for v in vertex_colors], dtype=[('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])

    vertex_all = np.empty(len(vertexs), vertexs.dtype.descr + vertex_colors.dtype.descr)
    for prop in vertexs.dtype.names:
        vertex_all[prop] = vertexs[prop]
    for prop in vertex_colors.dtype.names:
        vertex_all[prop] = vertex_colors[prop]

    el = PlyElement.describe(vertex_all, 'vertex')
    PlyData([el]).write(plyfilename)
    print("saving the final model to", plyfilename)


if __name__ == '__main__':
    # step1. save all the depth maps and the masks in outputs directory
    save_depth()
    img_wh = (1920, 1056)

    # intermediate dataset
    if args.split == "intermediate":
        scans = ['Family', 'Francis', 'Horse', 'Lighthouse', 'M60', 'Panther', 'Playground', 'Train']
        image_sizes = {'Family': (1920, 1080),
                       'Francis': (1920, 1080),
                       'Horse': (1920, 1080),
                       'Lighthouse': (2048, 1080),
                       'M60': (2048, 1080),
                       'Panther': (2048, 1080),
                       'Playground': (1920, 1080),
                       'Train': (1920, 1080)}
        geo_mask_thres = {'Family': 5,
                          'Francis': 6,
                          'Horse': 4,
                          'Lighthouse': 6,
                          'M60': 5,
                          'Panther': 5,
                          'Playground': 6,
                          'Train': 5}
        photo_thres = {'Family': 0.8,
                       'Francis': 0.8,
                       'Horse': 0.6,
                       'Lighthouse': 0.8,
                       'M60': 0.8,
                       'Panther': 0.8,
                       'Playground': 0.8,
                       'Train': 0.8}
        for scan in scans:
            scan_folder = os.path.join(args.testpath, args.split, scan)
            out_folder = os.path.join(args.outdir, scan)
            # step2. filter saved depth maps with geometric constraints
            filter_depth(
                scan_folder, out_folder, os.path.join(args.outdir, scan + '.ply'), args.geo_pixel_thres,
                args.geo_depth_thres, photo_thres[scan], img_wh, image_sizes[scan], geo_mask_thres[scan])
    # advanced dataset
    elif args.split == "advanced":
        scans = ['Auditorium', 'Ballroom', 'Courtroom', 'Museum', 'Palace', 'Temple']
        image_sizes = {'Auditorium': (1920, 1080),
                       'Ballroom': (1920, 1080),
                       'Courtroom': (1920, 1080),
                       'Museum': (1920, 1080),
                       'Palace': (1920, 1080),
                       'Temple': (1920, 1080)}
        geo_mask_thres = {'Auditorium': 3,
                          'Ballroom': 4,
                          'Courtroom': 4,
                          'Museum': 4,
                          'Palace': 5,
                          'Temple': 4}
        photo_thres = {'Auditorium': 0.8,
                       'Ballroom': 0.8,
                       'Courtroom': 0.8,
                       'Museum': 0.8,
                       'Palace': 0.8,
                       'Temple': 0.8}
        for scan in scans:
            scan_folder = os.path.join(args.testpath, args.split, scan)
            out_folder = os.path.join(args.outdir, scan)
            # step2. filter saved depth maps with geometric constraints
            filter_depth(
                scan_folder, out_folder, os.path.join(args.outdir, scan + '.ply'), args.geo_pixel_thres,
                args.geo_depth_thres, photo_thres[scan], img_wh, image_sizes[scan], geo_mask_thres[scan])
