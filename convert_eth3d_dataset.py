import argparse
import os
import shutil
from datasets.data_io import read_image_dictionary

from datasets.data_io import read_map, save_image

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert ETH 3D training dataset to standard input format")
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

        input_cam_path = os.path.join(args.input_folder, scan, "cams")
        # Read image index file
        image_index = read_image_dictionary(os.path.join(input_cam_path, "index2prefix.txt"))

        # Copy pair file
        shutil.copy(os.path.join(input_cam_path, "pair.txt"), os.path.join(scan_path, "pair.txt"))

        for cam_file in os.listdir(input_cam_path):
            if cam_file == "index2prefix.txt" or cam_file == "pair.txt":
                continue

            # Extract view ID and write cam file
            view_id = int(cam_file.split("_")[0])
            shutil.copy(os.path.join(input_cam_path, cam_file), os.path.join(cam_path, cam_file))

            # Copy image
            image_filename = os.path.join(args.input_folder, scan, "images", image_index[view_id])
            shutil.copy(image_filename, os.path.join(image_path, "{:0>8}.png".format(view_id)))

            # Copy GT depth map
            depth_gt_filename = os.path.join(args.input_folder, scan, "depths", image_index[view_id])
            depth_gt_filename = os.path.splitext(depth_gt_filename.replace("_undistorted", ""))[0] + ".pfm"
            shutil.copy(depth_gt_filename, os.path.join(depth_path, "{:0>8}.pfm".format(view_id)))

            # Create mask from GT depth map and save in output
            mask = (read_map(depth_gt_filename) > 0.0).squeeze(2).astype(bool)
            save_image(os.path.join(mask_path, "{:0>8}.png".format(view_id)), mask)
