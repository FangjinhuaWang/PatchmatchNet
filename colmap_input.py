import argparse
import cv2
import numpy as np
import os
import shutil
import struct
from typing import Dict, List, NamedTuple, Tuple


# ============================ read_model.py ============================#
class CameraModel(NamedTuple):
    model_id: int
    model_name: str
    num_params: int


class Camera(NamedTuple):
    id: int
    model: str
    width: int
    height: int
    params: List[float]


class Image(NamedTuple):
    id: int
    qvec: List[float]
    tvec: List[float]
    camera_id: int
    name: str
    point3d_ids: List[int] = []


class Point3D(NamedTuple):
    id: int
    xyz: List[float]
    rgb: List[int]
    error: float
    image_ids: List[int]
    point2d_ids: List[int]


CAMERA_MODELS = {
    CameraModel(0, "SIMPLE_PINHOLE", 3),
    CameraModel(1, "PINHOLE", 4),
    CameraModel(2, "SIMPLE_RADIAL", 4),
    CameraModel(3, "RADIAL", 5),
    CameraModel(4, "OPENCV", 8),
    CameraModel(5, "OPENCV_FISHEYE", 8),
    CameraModel(6, "FULL_OPENCV", 12),
    CameraModel(7, "FOV", 5),
    CameraModel(8, "SIMPLE_RADIAL_FISHEYE", 4),
    CameraModel(9, "RADIAL_FISHEYE", 5),
    CameraModel(10, "THIN_PRISM_FISHEYE", 12)
}
CAMERA_MODEL_IDS = dict([(camera_model.model_id, camera_model) for camera_model in CAMERA_MODELS])


def read_next_bytes(fid, num_bytes: int, format_char_sequence: str) -> Tuple:
    """Read and unpack the next bytes from a binary file.
    :param fid:
    :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    :return: Tuple of read and unpacked values.
    """
    data = fid.read(num_bytes)
    return struct.unpack("<" + format_char_sequence, data)


def read_cameras_text(path: str) -> Dict[int, Camera]:
    """
    see: src/base/reconstruction.cc
        void Reconstruction::WriteCamerasText(const std::string& path)
        void Reconstruction::ReadCamerasText(const std::string& path)
    """
    model_cameras: Dict[int, Camera] = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elements = line.split()
                cam_id = int(elements[0])
                model = elements[1]
                width = int(elements[2])
                height = int(elements[3])
                params = list(map(float, elements[4:]))
                model_cameras[cam_id] = Camera(cam_id, model, width, height, params)
    return model_cameras


def read_cameras_binary(path: str) -> Dict[int, Camera]:
    """
    see: src/base/reconstruction.cc
        void Reconstruction::WriteCamerasBinary(const std::string& path)
        void Reconstruction::ReadCamerasBinary(const std::string& path)
    """
    model_cameras: Dict[int, Camera] = {}
    with open(path, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        print("num of cameras")
        print(num_cameras)
        for camera_line_index in range(num_cameras):
            camera_properties = read_next_bytes(fid, 24, "iiQQ")
            cam_id = camera_properties[0]
            print("camera id")
            print(cam_id)

            model_id = camera_properties[1]
            model_name = CAMERA_MODEL_IDS[camera_properties[1]].model_name
            width = camera_properties[2]
            height = camera_properties[3]
            num_params = CAMERA_MODEL_IDS[model_id].num_params
            params = list(read_next_bytes(fid, 8 * num_params, "d" * num_params))
            model_cameras[cam_id] = Camera(cam_id, model_name, width, height, params)
        assert len(model_cameras) == num_cameras
    return model_cameras


def read_images_text(path: str) -> List[Image]:
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadImagesText(const std::string& path)
        void Reconstruction::WriteImagesText(const std::string& path)
    """
    model_images: List[Image] = []
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elements = line.split()
                im_id = int(elements[0])
                qvec = list(map(float, elements[1:5]))
                tvec = list(map(float, elements[5:8]))
                cam_id = int(elements[8])
                image_name = elements[9]
                elements = fid.readline().split()
                point3d_ids = list(map(int, elements[2::3]))
                model_images.append(Image(im_id, qvec, tvec, cam_id, image_name, point3d_ids))
    return model_images


def read_images_binary(path: str) -> List[Image]:
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadImagesBinary(const std::string& path)
        void Reconstruction::WriteImagesBinary(const std::string& path)
    """
    model_images: List[Image] = []
    with open(path, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for image_index in range(num_reg_images):
            binary_image_properties = read_next_bytes(fid, 64, "idddddddi")
            im_id = binary_image_properties[0]
            qvec = binary_image_properties[1:5]
            tvec = binary_image_properties[5:8]
            cam_id = binary_image_properties[8]
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":  # look for the ASCII 0 entry
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
            num_points_2d = read_next_bytes(fid, 8, "Q")[0]
            x_y_id_s = read_next_bytes(fid, 24 * num_points_2d, "ddq" * num_points_2d)
            point3d_ids = list(map(int, x_y_id_s[2::3]))
            model_images.append(Image(im_id, qvec, tvec, cam_id, image_name, point3d_ids))
    return model_images


def read_points_3d_text(path: str) -> Dict[int, Point3D]:
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DText(const std::string& path)
        void Reconstruction::WritePoints3DText(const std::string& path)
    """
    model_points3d: Dict[int, Point3D] = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elements = line.split()
                point_id = int(elements[0])
                xyz = list(map(float, elements[1:4]))
                rgb = list(map(int, elements[4:7]))
                error = float(elements[7])
                image_ids = list(map(int, elements[8::2]))
                point2d_ids = list(map(int, elements[9::2]))
                model_points3d[point_id] = Point3D(point_id, xyz, rgb, error, image_ids, point2d_ids)
    return model_points3d


def read_points3d_binary(path: str) -> Dict[int, Point3D]:
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DBinary(const std::string& path)
        void Reconstruction::WritePoints3DBinary(const std::string& path)
    """
    model_points3d: Dict[int, Point3D] = {}
    with open(path, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]
        for point_line_index in range(num_points):
            binary_point_line_properties = read_next_bytes(fid, 43, "QdddBBBd")
            point_id = binary_point_line_properties[0]
            xyz = list(binary_point_line_properties[1:4])
            rgb = list(binary_point_line_properties[4:7])
            error = binary_point_line_properties[7]
            track_length = read_next_bytes(fid, 8, "Q")[0]
            track_elements = read_next_bytes(fid, 8 * track_length, "ii" * track_length)
            image_ids = list(map(int, track_elements[0::2]))
            point2d_ids = list(map(int, track_elements[1::2]))
            model_points3d[point_id] = Point3D(point_id, xyz, rgb, error, image_ids, point2d_ids)
    return model_points3d


def read_model(path: str, ext: str) -> Tuple[Dict[int, Camera], List[Image], Dict[int, Point3D]]:
    if ext == ".txt":
        model_cameras = read_cameras_text(os.path.join(path, "cameras" + ext))
        model_images = read_images_text(os.path.join(path, "images" + ext))
        model_points_3d = read_points_3d_text(os.path.join(path, "points3D") + ext)
    else:
        model_cameras = read_cameras_binary(os.path.join(path, "cameras" + ext))
        model_images = read_images_binary(os.path.join(path, "images" + ext))
        model_points_3d = read_points3d_binary(os.path.join(path, "points3D") + ext)
    return model_cameras, model_images, model_points_3d


def quaternion_to_rotation_matrix(qvec: List[float]) -> np.ndarray:
    return np.array([
        [1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2]])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert colmap results into input for PatchmatchNet")

    parser.add_argument("--input_folder", type=str, help="Project input dir.")
    parser.add_argument("--output_folder", type=str, default="", help="Project output dir.")
    parser.add_argument("--num_src_images", type=int, default=-1, help="Related images")
    parser.add_argument("--theta0", type=float, default=5)
    parser.add_argument("--sigma1", type=float, default=1)
    parser.add_argument("--sigma2", type=float, default=10)
    parser.add_argument("--convert_format", action="store_true", default=False,
                        help="If set, convert image to jpg format.")

    args = parser.parse_args()

    if not args.output_folder:
        args.output_folder = args.input_folder

    if args.input_folder is None or not os.path.isdir(args.input_folder):
        raise Exception("Invalid input folder")

    if args.output_folder is None or not os.path.isdir(args.output_folder):
        raise Exception("Invalid output folder")

    image_dir = os.path.join(args.input_folder, "images")
    model_dir = os.path.join(args.input_folder, "sparse")
    cam_dir = os.path.join(args.output_folder, "cams")
    renamed_dir = os.path.join(args.output_folder, "images")

    cameras, images, points3d = read_model(model_dir, ".bin")
    num_images = len(images)

    param_type: Dict[str, List[str]] = {
        "SIMPLE_PINHOLE": ["f", "cx", "cy"],
        "PINHOLE": ["fx", "fy", "cx", "cy"],
        "SIMPLE_RADIAL": ["f", "cx", "cy", "k"],
        "SIMPLE_RADIAL_FISHEYE": ["f", "cx", "cy", "k"],
        "RADIAL": ["f", "cx", "cy", "k1", "k2"],
        "RADIAL_FISHEYE": ["f", "cx", "cy", "k1", "k2"],
        "OPENCV": ["fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2"],
        "OPENCV_FISHEYE": ["fx", "fy", "cx", "cy", "k1", "k2", "k3", "k4"],
        "FULL_OPENCV": ["fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6"],
        "FOV": ["fx", "fy", "cx", "cy", "omega"],
        "THIN_PRISM_FISHEYE": ["fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3", "k4", "sx1", "sy1"]
    }

    # intrinsic
    intrinsic: Dict[int, np.ndarray] = {}
    for camera_id, cam in cameras.items():
        params_dict = {key: value for key, value in zip(param_type[cam.model], cam.params)}
        if "f" in param_type[cam.model]:
            params_dict["fx"] = params_dict["f"]
            params_dict["fy"] = params_dict["f"]
        i = np.array([
            [params_dict["fx"], 0, params_dict["cx"]],
            [0, params_dict["fy"], params_dict["cy"]],
            [0, 0, 1]
        ])
        intrinsic[camera_id] = i
    print("intrinsic[1]\n", intrinsic[1], end="\n\n")

    # extrinsic
    extrinsic: List[np.ndarray] = []
    for i in range(num_images):
        e = np.zeros((4, 4))
        e[:3, :3] = quaternion_to_rotation_matrix(images[i].qvec)
        e[:3, 3] = images[i].tvec
        e[3, 3] = 1
        extrinsic.append(e)
    print("extrinsic[0]\n", extrinsic[0], end="\n\n")

    # depth range and interval
    depth_ranges: List[Tuple[float, float]] = []
    for i in range(num_images):
        zs = []
        for p3d_id in images[i].point3d_ids:
            if p3d_id == -1:
                continue
            transformed: np.ndarray = np.matmul(
                extrinsic[i], [points3d[p3d_id].xyz[0], points3d[p3d_id].xyz[1], points3d[p3d_id].xyz[2], 1])
            zs.append(transformed[2].item())
        zs_sorted = sorted(zs)
        # relaxed depth range
        depth_min = zs_sorted[int(len(zs) * .01)]
        depth_max = zs_sorted[int(len(zs) * .99)]

        depth_ranges.append((depth_min, depth_max))
    print("depth_ranges[0]\n", depth_ranges[0], end="\n\n")

    def calc_score(ind1: int, ind2: int) -> float:
        id_i = images[ind1].point3d_ids
        id_j = images[ind2].point3d_ids
        id_intersect = [it for it in id_i if it in id_j]
        cam_center_i = -np.matmul(extrinsic[ind1][:3, :3].transpose(), extrinsic[ind1][:3, 3:4])[:, 0]
        cam_center_j = -np.matmul(extrinsic[ind2][:3, :3].transpose(), extrinsic[ind2][:3, 3:4])[:, 0]
        view_score_ = 0.0
        for pid in id_intersect:
            if pid == -1:
                continue
            p = points3d[pid].xyz
            theta = (180 / np.pi) * np.arccos(
                np.dot(cam_center_i - p, cam_center_j - p) / np.linalg.norm(cam_center_i - p) / np.linalg.norm(
                    cam_center_j - p))
            view_score_ += np.exp(-(theta - args.theta0) * (theta - args.theta0) / (
                    2 * (args.sigma1 if theta <= args.theta0 else args.sigma2) ** 2))
        return view_score_

    # view selection
    score = np.zeros((num_images, num_images))
    queue: List[Tuple[int, int]] = []
    for i in range(num_images):
        for j in range(i + 1, num_images):
            queue.append((i, j))

    for i, j in queue:
        s = calc_score(i, j)
        score[i, j] = s
        score[j, i] = s

    if args.num_src_images < 0:
        args.num_src_images = num_images

    view_sel: List[List[Tuple[int, float]]] = []
    for i in range(num_images):
        sorted_score = np.argsort(score[i])[::-1]
        view_sel.append([(k, score[i, k]) for k in sorted_score[:args.num_src_images]])
    print("view_sel[0]\n", view_sel[0], end="\n\n")

    # write
    os.makedirs(cam_dir, exist_ok=True)
    os.makedirs(renamed_dir, exist_ok=True)
    for i in range(num_images):
        with open(os.path.join(cam_dir, "%08d_cam.txt" % i), "w") as f:
            f.write("extrinsic\n")
            for j in range(4):
                for k in range(4):
                    f.write(str(extrinsic[i][j, k]) + " ")
                f.write("\n")
            f.write("\nintrinsic\n")
            for j in range(3):
                for k in range(3):
                    f.write(str(intrinsic[images[i].camera_id][j, k]) + " ")
                f.write("\n")
            f.write("\n%f %f \n" % (depth_ranges[i][0], depth_ranges[i][1]))

    with open(os.path.join(args.output_folder, "pair.txt"), "w") as f:
        f.write("%d\n" % len(images))
        for i, sorted_score in enumerate(view_sel):
            f.write("%d\n%d " % (i, len(sorted_score)))
            for image_id, s in sorted_score:
                f.write("%d %f " % (image_id, s))
            f.write("\n")

    for i in range(num_images):
        if args.convert_format:
            img = cv2.imread(os.path.join(image_dir, images[i].name))
            cv2.imwrite(os.path.join(renamed_dir, "%08d.jpg" % i), img)
        else:
            shutil.copyfile(os.path.join(image_dir, images[i].name),
                            os.path.join(renamed_dir, "%08d.jpg" % i))
