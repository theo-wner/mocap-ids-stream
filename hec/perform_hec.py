import subprocess
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R

def run_colmap(dataset_path):
    feature_extractor = f"colmap feature_extractor --database_path {dataset_path}/database.db --image_path {dataset_path}/images"
    exhaustive_matcher = f"colmap exhaustive_matcher --database_path {dataset_path}/database.db"
    mkdir_1 = f"mkdir -p {dataset_path}/sparse"
    mapper = f"colmap mapper --database_path {dataset_path}/database.db --image_path {dataset_path}/images --output_path {dataset_path}/sparse"
    mkdir_2 = f"mkdir -p {dataset_path}/sparse_txt"
    model_converter = f"colmap model_converter --input_path {dataset_path}/sparse/0 --output_path {dataset_path}/sparse_txt --output_type TXT"

    cmd = f"{feature_extractor} && {exhaustive_matcher} && {mkdir_1} && {mapper} && {mkdir_2} && {model_converter}"

    try:
        result = subprocess.run(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"Command failed with return code: {e.returncode}")
        print("Output:", e.output)

def extract_colmap_poses(images_txt_path, output_txt_path):
    with open(images_txt_path, 'r') as f:
        lines = [line.strip() for line in f if not line.startswith('#') and line.strip() != '']

    with open(output_txt_path, 'w') as out_file:
        # Write header
        out_file.write("IMAGE_ID QW QX QY QZ TX TY TZ IMAGE_NAME\n")

        # Process lines two by two (image metadata + points line)
        for i in range(0, len(lines), 2):
            meta_line = lines[i]

            parts = meta_line.split()
            if len(parts) < 10:
                continue  # skip malformed lines

            image_id = parts[0]
            # Format floats to 6 decimal places
            qw, qx, qy, qz = [f"{float(x):.6f}" for x in parts[1:5]]
            tx, ty, tz = [f"{float(x):.6f}" for x in parts[5:8]]
            image_name = parts[9]

            out_file.write(f"{image_id} {qw} {qx} {qy} {qz} {tx} {ty} {tz} {image_name}\n")

def read_poses(filename):
    poses = {}
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('#') or line.strip() == '' or line.startswith('IMAGE_ID'):
                continue
            parts = line.strip().split()
            image_id = parts[0]
            qw, qx, qy, qz = map(float, parts[1:5])
            tx, ty, tz = map(float, parts[5:8])
            rot = R.from_quat([qx, qy, qz, qw]).as_matrix()
            T = np.eye(4)
            T[:3, :3] = rot
            T[:3, 3] = [tx, ty, tz]
            poses[image_id] = T
    return poses

import numpy as np

def compute_scale_factor(mocap_poses, colmap_poses):
    # Use only shared keys
    common_keys = sorted(set(mocap_poses.keys()) & set(colmap_poses.keys()))

    # Extract translations
    mocap_t = np.array([mocap_poses[k][:3, 3] for k in common_keys])
    colmap_t = np.array([colmap_poses[k][:3, 3] for k in common_keys])

    # Center data
    mocap_centered = mocap_t - mocap_t.mean(axis=0)
    colmap_centered = colmap_t - colmap_t.mean(axis=0)

    # Compute norms (distances to center)
    mocap_norms = np.linalg.norm(mocap_centered, axis=1)
    colmap_norms = np.linalg.norm(colmap_centered, axis=1)

    # Compute combined sort metric — e.g., colmap distance
    # Sort by colmap_norms (but could also sort by mocap_norms or average)
    sorted_indices = np.argsort(colmap_norms)

    # Keep only the points furthest away
    num_keep = 5
    selected_indices = sorted_indices[-num_keep:]  # last half = farthest

    # Filter
    filtered_mocap_norms = mocap_norms[selected_indices]
    filtered_colmap_norms = colmap_norms[selected_indices]
    per_point_scales = filtered_mocap_norms / filtered_colmap_norms

    # Compute statistics
    scale = np.mean(per_point_scales)
    scale_std = np.std(per_point_scales)

    return scale, scale_std

def compute_relative_motions(poses):
    keys = sorted(poses.keys())
    rotations = []
    translations = []
    for i in range(len(keys) - 1):
        T_i = poses[keys[i]]
        T_j = poses[keys[i + 1]]
        rel = np.linalg.inv(T_i) @ T_j
        R_rel = rel[:3, :3]
        t_rel = rel[:3, 3]
        rotations.append(R_rel)
        translations.append(t_rel)
    return rotations, translations

def apply_scale_factor(mocap_poses, scale):
    scaled_poses = {}
    for key, T in mocap_poses.items():
        T_scaled = T.copy()
        T_scaled[:3, 3] *= scale
        scaled_poses[key] = T_scaled
    return scaled_poses


if __name__ == "__main__":
    # Example usage:
    #run_colmap(dataset_path = "./data/hec_colmap")
    #extract_colmap_poses(images_txt_path="./data/hec_colmap/sparse_txt/images.txt", output_txt_path="./data/hec_colmap/colmap_poses.txt")
    mocap_poses = read_poses("./data/hec_colmap/mocap_poses.txt")
    colmap_poses = read_poses("./data/hec_colmap/colmap_poses.txt")

    scale, scale_std = compute_scale_factor(mocap_poses, colmap_poses)
    scaled_colmap_poses = apply_scale_factor(colmap_poses, scale)

    R_A, t_A = compute_relative_motions(mocap_poses)
    R_B, t_B = compute_relative_motions(scaled_colmap_poses)

    R_X, t_X = cv2.calibrateHandEye(R_A, t_A, R_B, t_B, method=cv2.CALIB_HAND_EYE_TSAI)

    X = np.eye(4)
    X[:3, :3] = R_X
    X[:3, 3] = t_X.flatten()

    print("Hand-eye transform X (from mocap to camera):")
    print(X)
