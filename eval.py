import argparse
import os
import sys
import time
from typing import OrderedDict

import cv2
import numpy as np
import torch.backends.cudnn
import torch.nn as nn
import torch.nn.parallel
from plyfile import PlyData, PlyElement
from torch import Tensor
from torch.utils.data import DataLoader

from datasets.data_io import read_cam_file, read_image, read_map, read_pair_file, save_image, save_map
from datasets.mvs import MVSDataset
from models.net import PatchMatchNet
from utils import print_args, tensor2numpy, to_cuda


# run MVS model to save depth maps
def save_depth(args):
    if args.input_type == 'params':
        print('Evaluating model with params from {}'.format(args.checkpoint_path))
        model = PatchMatchNet(args.patch_match_interval_scale, args.propagation_range, args.patch_match_iteration,
                              args.patch_match_num_sample, args.propagate_neighbors, args.evaluate_neighbors, False)

        state_dict = torch.load(args.checkpoint_path)['model']

        if args.data_parallel:
            print('Data parallel mode')
            model = nn.DataParallel(model)
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
        else:
            print('Non-parallel mode')
            # For non data parallel mode we need to convert the state dictionary and remove the 'module.` prefix
            new_dict: OrderedDict[str, Tensor] = {}
            for key in state_dict:
                new_dict[key[7:]] = state_dict[key]
            missing, unexpected = model.load_state_dict(new_dict, strict=False)
        print('Missing keys')
        for key in missing:
            print(key)
        print('Unexpected keys')
        for key in unexpected:
            print(key)
    else:
        print('Using scripted module from {}'.format(args.checkpoint_path))
        model = torch.jit.load(args.checkpoint_path)

    model.cuda()
    model.eval()
    # sm = torch.jit.script(model)
    # sm.save(os.path.join(args.output_folder, 'patchmatchnet-module.pt'))
    # return

    # Setup dataset parameters based on type of dataset
    num_light_idx = -1
    cam_folder = 'cams'
    pair_path = 'pair.txt'
    image_folder = 'images'
    depth_folder = 'depth_gt'
    index_path = None
    if args.dataset_type == 'dtu':
        num_light_idx = 7
    elif args.dataset_type == 'eth3d':
        pair_path = 'cams/pair.txt'
        depth_folder = 'depths'
        index_path = 'cams/index2prefix.txt'
    elif args.dataset_type == 'blended':
        pair_path = 'cams/pair.txt'
        image_folder = 'blended_images'
        depth_folder = 'rendered_depth_maps'

    dataset = MVSDataset(args.input_folder, args.num_views, args.image_max_dim, args.scan_list, False, num_light_idx,
                         cam_folder, pair_path, image_folder, depth_folder, index_path)
    image_loader = DataLoader(dataset, args.batch_size, shuffle=False, num_workers=4, drop_last=False)

    with torch.no_grad():
        for batch_idx, sample in enumerate(image_loader):
            start_time = time.time()
            sample_cuda = to_cuda(sample)

            depth, conf, _ = model.forward(sample_cuda['images'], sample_cuda['intrinsics'], sample_cuda['extrinsics'],
                                           sample_cuda['depth_params'])

            depth = tensor2numpy(depth)
            conf = tensor2numpy(conf)
            del sample_cuda
            print('Iter {}/{}, time = {:.3f}'.format(batch_idx + 1, len(image_loader), time.time() - start_time))
            filenames = sample['filename']

            # save depth maps and confidence maps
            for filename, depth_est, photometric_confidence in zip(filenames, depth, conf):
                depth_filename = os.path.join(args.output_folder, filename.format('depth_est', args.file_format))
                confidence_filename = os.path.join(args.output_folder, filename.format('confidence', args.file_format))
                os.makedirs(os.path.dirname(depth_filename), exist_ok=True)
                os.makedirs(os.path.dirname(confidence_filename), exist_ok=True)
                # save depth maps
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


def check_geometric_consistency(depth_ref, intrinsics_ref, extrinsics_ref, depth_src, intrinsics_src, extrinsics_src):
    width, height = depth_ref.shape[1], depth_ref.shape[0]
    x_ref, y_ref = np.meshgrid(np.arange(0, width), np.arange(0, height))
    depth_reprojected, x2d_reprojected, y2d_reprojected, x2d_src, y2d_src = reproject_with_depth(depth_ref,
                                                                                                 intrinsics_ref,
                                                                                                 extrinsics_ref,
                                                                                                 depth_src,
                                                                                                 intrinsics_src,
                                                                                                 extrinsics_src)
    # print(depth_ref.shape)
    # print(depth_reprojected.shape)
    # check |p_reproject - p_1| < 1
    dist = np.sqrt((x2d_reprojected - x_ref) ** 2 + (y2d_reprojected - y_ref) ** 2)

    # check |d_reproject - d_1| / d_1 < 0.01
    # depth_ref = np.squeeze(depth_ref, 2)
    depth_diff = np.abs(depth_reprojected - depth_ref)
    relative_depth_diff = depth_diff / depth_ref

    mask = np.logical_and(dist < input_args.geom_pixel_threshold, relative_depth_diff < input_args.geom_depth_threshold)
    depth_reprojected[~mask] = 0

    return mask, depth_reprojected, x2d_src, y2d_src


def filter_depth(args):
    # the pair file
    pair_file = os.path.join(args.input_folder, 'pair.txt')
    # for the final point cloud
    vertices = []
    vertex_colors = []

    pair_data = read_pair_file(pair_file)

    # for each reference view and the corresponding source views
    for ref_view, src_views in pair_data:

        # load the reference image
        ref_img, original_h, original_w = read_image(
            os.path.join(args.input_folder, 'images/{:0>8}.jpg'.format(ref_view)), args.image_max_dim)
        ref_intrinsics, ref_extrinsics = read_cam_file(
            os.path.join(args.input_folder, 'cams/{:0>8}_cam.txt'.format(ref_view)))[0:2]
        ref_intrinsics[0] *= ref_img.shape[1] / original_w
        ref_intrinsics[1] *= ref_img.shape[0] / original_h

        # load the estimated depth of the reference view
        ref_depth_est = \
            read_map(os.path.join(args.output_folder, 'depth_est/{:0>8}{}'.format(ref_view, args.file_format)))
        ref_depth_est = np.squeeze(ref_depth_est, 2)
        # load the photometric mask of the reference view
        confidence = \
            read_map(os.path.join(args.output_folder, 'confidence/{:0>8}{}'.format(ref_view, args.file_format)))

        photo_mask = confidence > args.photo_threshold
        photo_mask = np.squeeze(photo_mask, 2)

        all_src_view_depth_estimates = []

        # compute the geometric mask
        geo_mask_sum = 0
        for src_view in src_views:
            # camera parameters of the source view
            src_image, original_h, original_w = read_image(
                os.path.join(args.input_folder, 'images/{:0>8}.jpg'.format(src_view)), args.image_max_dim)
            src_intrinsics, src_extrinsics = read_cam_file(
                os.path.join(args.input_folder, 'cams/{:0>8}_cam.txt'.format(src_view)))[0:2]
            src_intrinsics[0] *= src_image.shape[1] / original_w
            src_intrinsics[1] *= src_image.shape[0] / original_h

            # the estimated depth of the source view
            src_depth_est = \
                read_map(os.path.join(args.output_folder, 'depth_est/{:0>8}{}'.format(src_view, args.file_format)))

            geo_mask, depth_reprojected, x2d_src, y2d_src = \
                check_geometric_consistency(ref_depth_est, ref_intrinsics, ref_extrinsics, src_depth_est,
                                            src_intrinsics, src_extrinsics)
            geo_mask_sum += geo_mask.astype(np.int32)
            all_src_view_depth_estimates.append(depth_reprojected)

        depth_est_averaged = (sum(all_src_view_depth_estimates) + ref_depth_est) / (geo_mask_sum + 1)

        geo_mask = geo_mask_sum >= args.geom_threshold
        final_mask = np.logical_and(photo_mask, geo_mask)

        os.makedirs(os.path.join(args.output_folder, 'mask'), exist_ok=True)
        save_image(os.path.join(args.output_folder, 'mask/{:0>8}_photo.png'.format(ref_view)), photo_mask)
        save_image(os.path.join(args.output_folder, 'mask/{:0>8}_geo.png'.format(ref_view)), geo_mask)
        save_image(os.path.join(args.output_folder, 'mask/{:0>8}_final.png'.format(ref_view)), final_mask)

        print('processing {}, ref-view{:0>3}, geo_mask:{:3f} photo_mask:{:3f} final_mask: {:3f}'.format(
            args.input_folder, ref_view, geo_mask.mean(), photo_mask.mean(), final_mask.mean()))

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

        x, y, depth = x[valid_points], y[valid_points], depth_est_averaged[valid_points]

        color = ref_img[valid_points]
        xyz_ref = np.matmul(np.linalg.inv(ref_intrinsics),
                            np.vstack((x, y, np.ones_like(x))) * depth)
        xyz_world = np.matmul(np.linalg.inv(ref_extrinsics),
                              np.vstack((xyz_ref, np.ones_like(x))))[:3]
        vertices.append(xyz_world.transpose((1, 0)))
        vertex_colors.append((color * 255).astype(np.uint8))

    vertices = np.concatenate(vertices, axis=0)
    vertex_colors = np.concatenate(vertex_colors, axis=0)
    vertices = np.array([tuple(v) for v in vertices], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    vertex_colors = np.array([tuple(v) for v in vertex_colors], dtype=[('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])

    vertex_all = np.empty(len(vertices), vertices.dtype.descr + vertex_colors.dtype.descr)
    for prop in vertices.dtype.names:
        vertex_all[prop] = vertices[prop]
    for prop in vertex_colors.dtype.names:
        vertex_all[prop] = vertex_colors[prop]

    el = PlyElement.describe(vertex_all, 'vertex')
    ply_filename = os.path.join(args.output_folder, 'fused.ply')
    PlyData([el]).write(ply_filename)
    print('saving the final model to', ply_filename)


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True

    parser = argparse.ArgumentParser(description='Predict depth, filter, and fuse')

    # High level input/output options
    parser.add_argument('--input_folder', type=str, help='input data path')
    parser.add_argument('--output_folder', type=str, help='output path')
    parser.add_argument('--checkpoint_path', type=str, help='load a specific checkpoint for parameters of model')
    parser.add_argument('--file_format', type=str, default='.bin', help='File format for depth maps',
                        choices=['.bin', '.pfm'])
    parser.add_argument('--input_type', type=str, default='params', help='Input type of checkpoint',
                        choices=['params', 'module'])
    parser.add_argument('--output_type', type=str, default='both', help='Type of outputs to produce',
                        choices=['depth', 'fusion', 'both'])
    parser.add_argument('--data_parallel', type=bool, default=False, help='Flag to use or skip data parallel mode')

    # Dataset loading options
    parser.add_argument('--dataset_type', type=str, default='custom', help='Type of dataset to configure parameters',
                        choices=['custom', 'dtu', 'eth3d', 'blended'])
    parser.add_argument('--num_views', type=int, default=21,
                        help='total views for each patch-match problem including reference')
    parser.add_argument('--image_max_dim', type=int, default=1024, help='max image dimension')
    parser.add_argument('--scan_list', type=str, default='',
                        help='Optional scan list text file to identify input folders')
    parser.add_argument('--batch_size', type=int, default=1, help='evaluation batch size')

    # PatchMatchNet module options (only used when not loading from file)
    parser.add_argument('--patch_match_interval_scale', nargs='+', type=float, default=[0.005, 0.0125, 0.025],
                        help='normalized interval in inverse depth range to generate samples in local perturbation')
    parser.add_argument('--propagation_range', nargs='+', type=int, default=[6, 4, 2],
                        help='fixed offset of sampling points for propagation of patch match on stages 1,2,3')
    parser.add_argument('--patch_match_iteration', nargs='+', type=int, default=[1, 2, 2],
                        help='num of iteration of patch match on stages 1,2,3')
    parser.add_argument('--patch_match_num_sample', nargs='+', type=int, default=[8, 8, 16],
                        help='num of generated samples in local perturbation on stages 1,2,3')
    parser.add_argument('--propagate_neighbors', nargs='+', type=int, default=[0, 8, 16],
                        help='num of neighbors for adaptive propagation on stages 1,2,3')
    parser.add_argument('--evaluate_neighbors', nargs='+', type=int, default=[9, 9, 9],
                        help='num of neighbors for adaptive matching cost aggregation of adaptive evaluation on stages 1,2,3')

    # Stereo fusion options
    parser.add_argument('--display', action='store_true', help='display depth images and masks')
    parser.add_argument('--geom_pixel_threshold', type=float, default=1.0,
                        help='pixel threshold for geometric consistency filtering')
    parser.add_argument('--geom_depth_threshold', type=float, default=0.01,
                        help='depth threshold for geometric consistency filtering')
    parser.add_argument('--geom_threshold', type=int, default=5, help='threshold for geometric consistency filtering')
    parser.add_argument('--photo_threshold', type=float, default=0.5,
                        help='threshold for photometric consistency filtering')

    # parse arguments and check
    input_args = parser.parse_args()
    print('argv:', sys.argv[1:])
    print_args(input_args)

    # step1. save all the depth maps and the masks in outputs directory
    if input_args.output_type == 'depth' or input_args.output_type == 'both':
        save_depth(input_args)

    # step2. filter saved depth maps and reconstruct point cloud
    if input_args.output_type == 'fusion' or input_args.output_type == 'both':
        filter_depth(input_args)
