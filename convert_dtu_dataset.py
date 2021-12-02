import argparse
import numpy as np
import os
import shutil
from PIL import Image

from datasets.data_io import read_image, read_map, save_image, save_map

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert DTU training dataset to standard input format")
    parser.add_argument("--input_folder", type=str, help="Input training data")
    parser.add_argument("--output_folder", type=str, help="Output converted training data")
    parser.add_argument("--scan_list", type=str, help="Input scan list for conversion")
    args = parser.parse_args()

    if args.input_folder is None or not os.path.isdir(args.input_folder):
        raise Exception("Invalid input folder")

    if args.output_folder is None or not os.path.isdir(args.output_folder):
        raise Exception("Invalid output folder")

    if args.scan_list is None or not os.path.isfile(args.scan_list):
        raise Exception("Invalid input scan list")

    # Read input scan list and create output scan list
    with open(args.scan_list) as f:
        scans = [line.rstrip() for line in f.readlines()]

    # Process each input scan and copy files to output folders
    for scan in scans:
        # Create output folders
        scan_path = os.path.join(args.output_folder, scan)
        cam_path = os.path.join(scan_path, "cams")
        depth_path = os.path.join(scan_path, "depth_gt")
        image_path = os.path.join(scan_path, "images")
        mask_path = os.path.join(scan_path, "masks")
        os.makedirs(scan_path, exist_ok=True)
        os.makedirs(cam_path, exist_ok=True)
        os.makedirs(depth_path, exist_ok=True)
        os.makedirs(image_path, exist_ok=True)
        os.makedirs(mask_path, exist_ok=True)

        # Copy pair file
        shutil.copy(os.path.join(args.input_folder, "Cameras_1/pair.txt"), os.path.join(scan_path, "pair.txt"))

        for cam_file in os.listdir(os.path.join(args.input_folder, "Cameras_1/train")):
            # Extract view ID and write cam file
            view_id = int(cam_file.split("_")[0])

            # Modify the cam file intrinsics by factor of 4 to match the training image size
            with open(os.path.join(args.input_folder, "Cameras_1/train", cam_file)) as f:
                lines = [line.rstrip() for line in f.readlines()]
            tmp = np.fromstring(lines[7], dtype=np.float32, sep=" ") * 4.0
            lines[7] = "{} {} {}".format(tmp[0], tmp[1], tmp[2])
            tmp = np.fromstring(lines[8], dtype=np.float32, sep=" ") * 4.0
            lines[8] = "{} {} {}".format(tmp[0], tmp[1], tmp[2])
            with open(os.path.join(cam_path, cam_file), "w") as f:
                for line in lines:
                    f.write(line + "\n")

            # Copy GT depth map after resizing and cropping
            depth_map = read_map(os.path.join(
                args.input_folder, "Depths_raw", scan, "depth_map_{:0>4}.pfm".format(view_id)), 800)
            depth_map = depth_map[44:556, 80:720]
            save_map(os.path.join(depth_path, "{:0>8}.pfm".format(view_id)), depth_map)

            # Copy mask after resizing and cropping
            mask = read_image(os.path.join(
                args.input_folder, "Depths_raw", scan, "depth_visual_{:0>4}.png".format(view_id)), 800)[0]
            mask = mask[44:556, 80:720] > 0.04
            save_image(os.path.join(mask_path, "{:0>8}.png".format(view_id)), mask)

            for light_idx in range(7):
                # Copy images for each light index into separate sub-folders
                image_prefix_path = os.path.join(image_path, str(light_idx))
                os.makedirs(image_prefix_path, exist_ok=True)
                image = Image.open(os.path.join(
                    args.input_folder, "Rectified/{}_train/rect_{:0>3}_{}_r5000.png".format(
                        scan, view_id + 1, light_idx)))
                image.save(os.path.join(image_prefix_path, "{:0>8}.jpg".format(view_id)))
