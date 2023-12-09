import argparse
import glob
import json
import os
import re
import shutil
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import PIL
from PIL import Image
from torchvision import transforms

parser = argparse.ArgumentParser(description="preprocess postech dataset to sdfstudio dataset")

parser.add_argument("--input_path", dest="input_path", help="path to postech scene")
parser.set_defaults(im_name="NONE")

parser.add_argument("--output_path", dest="output_path", help="path to output")
parser.set_defaults(store_name="NONE")
parser.add_argument(
    "--type",
    dest="type",
    default="mono_prior",
    choices=["none", "mono_prior", "sensor_depth"],
    help="mono_prior to use monocular prior, sensor_depth to use depth captured with a depth sensor (gt depth)",
)
parser.add_argument("--cam-type", dest="cam_type", default="perspective", choices=["perspective", "fisheye"])
parser.add_argument("--scene-type", dest="scene_type", default="unbound", choices=["indoor", "object", "unbound"])
parser.add_argument("--copy", dest="copy", default=False, action='store_true')
parser.add_argument("--resize", dest="resize", default=False, action='store_true')
args = parser.parse_args()


def alphanum_key(s):
    """Turn a string into a list of string and number chunks.
    "z23a" -> ["z", 23, "a"]
    """
    return [int(x) if x.isdigit() else x for x in re.split("([0-9]+)", s)]

output_path = Path(args.output_path)  # EX) "data/custom/postech_scene"
input_path = Path(args.input_path)    # EX) "/home/user/dataset/postech/scans/scene"

output_path.mkdir(parents=True, exist_ok=True)

# load color
if args.cam_type == "perspective":
    camera_model = "OPENCV"
    color_path = input_path / "color"
    intrinsic_path = input_path / "intrinsic" / "intrinsic_color.txt"
elif args.cam_type == "fisheye":
    camera_model = "OPENCV_FISHEYE"
    color_path = input_path / "color_fisheye"
    intrinsic_path = input_path / "intrinsic" / "intrinsic_fisheye.txt"

exts=["jpg", "png"]
for ext in exts:
    color_paths = sorted(glob.glob(os.path.join(color_path, f"*.{ext}")), key=alphanum_key)
    if len(color_paths) > 0: break        

# load depth
depth_path = input_path / "depth_dense"
exts=["png", "npy"]
for ext in exts:
    depth_paths = sorted(glob.glob(os.path.join(depth_path, f"*.{ext}")), key=alphanum_key)
    if len(depth_paths) > 0: break     

if len(depth_paths) == 0:
    depth_paths = color_paths

H, W = cv2.imread(color_paths[0]).shape[:2]
print("\nH, W:", H, ',', W)

# load intrinsic
camera_intrinsic = np.loadtxt(intrinsic_path)
print("intrinsic:", camera_intrinsic)

# load pose
pose_path = input_path / "pose"
poses = []
pose_paths = sorted(glob.glob(os.path.join(pose_path, "*.txt")), key=lambda x: int(os.path.basename(x)[:-4]))
for pose_path in pose_paths:
    c2w = np.loadtxt(pose_path)
    poses.append(c2w)
poses = np.array(poses)

# deal with invalid poses
valid_poses = np.isfinite(poses).all(axis=2).all(axis=1)
min_vertices = poses[:, :3, 3][valid_poses].min(axis=0)
max_vertices = poses[:, :3, 3][valid_poses].max(axis=0)

# OpenGL/Blender convention, needs to change to COLMAP/OpenCV convention
# https://docs.nerf.studio/en/latest/quickstart/data_conventions.html
# poses[:, 0:3, 1:3] *= -1

center = (min_vertices + max_vertices) / 2.0
scale = 2.0 / (np.max(max_vertices - min_vertices) + 3.0)
# scale = 3.0 / (np.max(max_vertices - min_vertices))
print("center, scale:", center, ',', scale)

# we should normalize pose to unit cube
poses[:, :3, 3] -= center
poses[:, :3, 3] *= scale

# inverse normalization
scale_mat = np.eye(4).astype(np.float32)
scale_mat[:3, 3] -= center
scale_mat[:3] *= scale
scale_mat = np.linalg.inv(scale_mat)

if args.resize is True:
    # center copy image if use monocular prior because omnidata use 384x384 as inputs
    # get smallest side to generate square crop
    target_crop = min(H, W)

    target_size = 1152
    trans_totensor = transforms.Compose(
        [
            transforms.CenterCrop(target_crop),
            transforms.Resize(target_size, interpolation=PIL.Image.BILINEAR),
        ]
    )
    depth_trans_totensor = transforms.Compose(
        [
            transforms.CenterCrop(target_crop),
            transforms.Resize(target_size, interpolation=PIL.Image.NEAREST),
        ]
    )

    # center crop by min_dim
    offset_x = (W - target_crop) * 0.5
    offset_y = (H - target_crop) * 0.5
    print(offset_x, offset_y)

    camera_intrinsic[0, 2] -= offset_x
    camera_intrinsic[1, 2] -= offset_y
    print(camera_intrinsic)

    # resize from min_dim x min_dim -> to 384 x 384
    resize_factor = target_size / target_crop
    print(resize_factor)

    camera_intrinsic[:2, :] *= resize_factor
    print(camera_intrinsic)

    # new H, W after center crop
    H, W = target_size, target_size

K = camera_intrinsic

frames = []
out_idx = 0

for idx, (valid, pose, image_path, depth_path) in enumerate(zip(valid_poses, poses, color_paths, depth_paths)):
    # if idx % 6 != 0 and idx % 6 != 3:
    #     continue
    
    if not valid:
        continue

    target_image = output_path / f"{out_idx:06d}_rgb.png"
    print(target_image)

    if args.copy is True:
        if args.resize is True:
            img = Image.open(image_path)
            img_tensor = trans_totensor(img)
            img_tensor.save(target_image)
        else:
            shutil.copyfile(image_path, target_image)

    rgb_path = str(target_image.relative_to(output_path))
    frame = {
        "rgb_path": rgb_path,
        "camtoworld": pose.tolist(),
        "intrinsics": K.tolist(),
    }
    
    if args.type == "mono_prior":
        frame.update(
            {
                "mono_depth_path": rgb_path.replace("_rgb.png", "_depth.npy"),
                "mono_normal_path": rgb_path.replace("_rgb.png", "_normal.npy"),
            }
        )
    elif args.type == "sensor_depth":
        frame["sensor_depth_path"] = rgb_path.replace("_rgb.png", "_depth.npy")
        target_image = output_path / frame["sensor_depth_path"]
        print(target_image)

        if args.copy is True:
            img_type = depth_path[-3:]

            # Convert depth to meters, then to "network units"
            depth_shift = 1000.0
            if img_type == 'npy':
                depth_map = np.load(depth_path) / depth_shift
            elif img_type == 'png':
                depth_map = cv2.imread(depth_path, -1).astype(np.float32) / depth_shift

            if args.resize is True:
                depth_PIL = Image.fromarray(depth_map)
                rz_depth_map = depth_trans_totensor(depth_PIL)
                depth_map = np.asarray(rz_depth_map)

            # Scale depth as we normalize the scene to unit box
            depth_map = depth_map * scale
            np.save(output_path / frame["sensor_depth_path"], depth_map)
            # Color map gt depth for visualization
            plt.imsave(output_path / frame["sensor_depth_path"].replace(".npy", ".png"), depth_map, cmap="viridis")

    frames.append(frame)
    out_idx += 1

# construct the scene box
if args.scene_type == "indoor":
    scene_box = {
        "aabb": [[-1, -1, -1], [1, 1, 1]],
        "near": 0.05,
        "far": 2.5,
        "radius": 1.0,
        "collider_type": "box",
    }
elif args.scene_type == "object":
    scene_box = {
        "aabb": [[-1, -1, -1], [1, 1, 1]],
        "near": 0.05,
        "far": 2.0,
        "radius": 1.0,
        "collider_type": "near_far",
    }
elif args.scene_type == "unbound":
    # TODO: case-by-case near far based on depth prior
    #  such as colmap sparse points or sensor depths
    scene_box = {
        "aabb": [min_vertices.tolist(), max_vertices.tolist()],
        "near": 0.05,
        "far": 2.5 * np.max(max_vertices - min_vertices),
        "radius": np.min(max_vertices - min_vertices) / 2.0,
        "collider_type": "box",
    }

# meta data
output_data = {
    "camera_model": camera_model,
    "height": H,
    "width": W,
    "has_mono_prior": args.type == "mono_prior",
    "has_sensor_depth": args.type == "sensor_depth",
    "pairs": None,
    "worldtogt": scale_mat.tolist(),
    "scene_box": scene_box,
}

output_data["frames"] = frames

# save as json
with open(output_path / "meta_data.json", "w", encoding="utf-8") as f:
    json.dump(output_data, f, indent=4)
