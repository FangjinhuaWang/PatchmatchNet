import argparse
import os
import shutil
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image as PilImage

from colmap_input import Camera, Image
from datasets.data_io import read_cam_file, read_map, read_pair_file, save_map


def rotation_matrix_to_quaternion(rot: np.ndarray) -> List[float]:
    rxx, ryx, rzx, rxy, ryy, rzy, rxz, ryz, rzz = rot.flat
    k = np.array([
        [rxx - ryy - rzz, 0, 0, 0],
        [ryx + rxy, ryy - rxx - rzz, 0, 0],
        [rzx + rxz, rzy + ryz, rzz - rxx - ryy, 0],
        [ryz - rzy, rzx - rxz, rxy - ryx, rxx + ryy + rzz]]) / 3.0
    eigenvalues, eigenvectors = np.linalg.eigh(k)
    qvec = eigenvectors[[3, 0, 1, 2], np.argmax(eigenvalues)]
    if qvec[0] < 0:
        qvec *= -1
    return [qvec[0], qvec[1], qvec[2], qvec[3]]


def create_output_dirs(path: str):
    os.makedirs(path, exist_ok=True)
    os.makedirs(os.path.join(path, 'images'), exist_ok=True)
    os.makedirs(os.path.join(path, 'sparse'), exist_ok=True)
    os.makedirs(os.path.join(path, 'stereo'), exist_ok=True)
    os.makedirs(os.path.join(path, 'stereo', 'confidence_maps'), exist_ok=True)
    os.makedirs(os.path.join(path, 'stereo', 'consistency_graphs'), exist_ok=True)
    os.makedirs(os.path.join(path, 'stereo', 'depth_maps'), exist_ok=True)
    os.makedirs(os.path.join(path, 'stereo', 'normal_maps'), exist_ok=True)


def copy_maps(input_path: str, output_path: str):
    shutil.copytree(os.path.join(input_path, 'images'), os.path.join(output_path, 'images'), dirs_exist_ok=True)
    output_type = os.path.splitext(os.listdir(os.path.join(input_path, 'depth_est'))[0])[1]
    if output_type == '.bin':
        shutil.copytree(os.path.join(input_path, 'depth_est'), os.path.join(output_path, 'stereo/depth_maps'), dirs_exist_ok=True)
        shutil.copytree(os.path.join(input_path, 'confidence'), os.path.join(output_path, 'stereo/confidence_maps'), dirs_exist_ok=True)
    else:
        # If output is .pfm we need to read and convert to colmap .bin
        for map_file in os.listdir(os.path.join(input_path, 'depth_est')):
            name, ext = os.path.splitext(map_file)
            save_map(os.path.join(output_path, 'stereo/depth_maps', name + '.bin'),
                     read_map(os.path.join(input_path, 'depth_est', map_file)))
            save_map(os.path.join(output_path, 'stereo/confidence_maps', name + '.bin'),
                     read_map(os.path.join(input_path, 'confidence', map_file)))


def read_reconstruction(path: str) -> Tuple[List[Camera], List[Image], List[Tuple[int, List[int]]]]:
    cameras = []
    images = []
    for cam_file in os.listdir(os.path.join(path, 'cams')):
        im_id = int(cam_file.split('_')[0])
        im_file = cam_file.split('_')[0] + '.jpg'
        image = PilImage.open(os.path.join(path, 'images', im_file), 'r')
        intrinsics, extrinsics, _ = read_cam_file(os.path.join(path, 'cams', cam_file))
        cameras.append(Camera(im_id, 'PINHOLE', image.width, image.height,
                              [intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]]))
        qvec = rotation_matrix_to_quaternion(extrinsics[0:3, 0:3])
        tvec = list(extrinsics[0:3, 3])
        images.append(Image(im_id, qvec, tvec, im_id, im_file))

    return cameras, images, read_pair_file(os.path.join(path, 'pair.txt'))


def write_config(path: str, images: List[Image], pairs: List[Tuple[int, List[int]]]):
    image_names: Dict[int, str] = {image.id: image.name for image in images}
    with open(path, 'w') as f:
        f.writelines([','.join([image_names[view_id] for view_id in [pair[0]] + pair[1]]) + '\n' for pair in pairs])


def write_sparse(path: str, cameras: List[Camera], images: List[Image]):
    write_cameras(os.path.join(path, 'cameras.txt'), cameras)
    write_images(os.path.join(path, 'images.txt'), images)
    write_points_3d(os.path.join(path, 'points3D.txt'))


# Write cameras in colmap text format
def write_cameras(path: str, cameras: List[Camera]):
    cam_list = ['{} {} {} {} {} {} {} {}\n'.format(c.id, c.model, c.width, c.height, c.params[0], c.params[1],
                                                   c.params[2], c.params[3]) for c in cameras]
    with open(path, 'w') as f:
        f.write('# Camera list with one line of data per camera:\n')
        f.write('#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n')
        f.write('# Number of cameras: {}\n'.format(len(cameras)))
        f.writelines(cam_list)


# Write images in colmap text format
def write_images(path: str, images: List[Image]):
    image_list = ['{} {} {} {} {} {} {} {} {} {}\n\n'.format(i.id, i.qvec[0], i.qvec[1], i.qvec[2], i.qvec[3], i.tvec[0],
                                                             i.tvec[1], i.tvec[2], i.camera_id, i.name) for i in images]
    with open(path, 'w') as f:
        f.write('# Image list with two lines of data per image:\n')
        f.write('#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n')
        f.write('#   POINTS2D[] as (X, Y, POINT3D_ID)\n')
        f.write('# Number of images: {}, mean observations per image: 0\n'.format(len(image_list)))
        f.writelines(image_list)


# This is a dummy write of an empty points3D file since we do not have actual sparse 3D points to use
def write_points_3d(path: str):
    with open(path, 'w') as f:
        f.write('# 3D point list with one line of data per point:\n')
        f.write('#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n')
        f.write('# Number of points: 0, mean track length: 0')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert colmap results into input for PatchMatchNet')

    parser.add_argument('--input_folder', type=str, help='Input PatchMatchNet reconstruction dir')
    parser.add_argument('--output_folder', type=str, default='', help='Output ColMap MVS workspace')

    args = parser.parse_args()

    if not args.output_folder:
        args.output_folder = args.input_folder

    create_output_dirs(args.output_folder)
    copy_maps(args.input_folder, args.output_folder)
    cams, ims, im_pairs = read_reconstruction(args.input_folder)
    write_config(os.path.join(args.output_folder, 'stereo/fusion.cfg'), ims, im_pairs)
    write_sparse(os.path.join(args.output_folder, 'sparse'), cams, ims)
