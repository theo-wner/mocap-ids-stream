import subprocess
import numpy as np

def run_colmap(dataset_path):
    """
    Runs the COLMAP-Pipeline on a dataset. The specified dataset_path has to contain a subfolder called images.
    Results in adding "colmap_poses.txt" to the dataset_path directory
    """
    feature_extractor = f"colmap feature_extractor --database_path {dataset_path}/database.db --image_path {dataset_path}/images"
    exhaustive_matcher = f"colmap exhaustive_matcher --database_path {dataset_path}/database.db"
    mkdir_1 = f"mkdir -p {dataset_path}/sparse"
    mapper = f"colmap mapper --database_path {dataset_path}/database.db --image_path {dataset_path}/images --output_path {dataset_path}/sparse"
    mkdir_2 = f"mkdir -p {dataset_path}/sparse_txt"
    model_converter = f"colmap model_converter --input_path {dataset_path}/sparse/0 --output_path {dataset_path}/sparse_txt --output_type TXT"

    cmd = f"{feature_extractor} && {exhaustive_matcher} && {mkdir_1} && {mapper} && {mkdir_2} && {model_converter}"

    process = subprocess.Popen(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    for line in process.stdout:
        print(line, end="")
    process.wait() 

    # Extract only the colmap poses
    with open(f"{dataset_path}/sparse_txt/images.txt", 'r') as f:
        lines = [line.strip() for line in f if not line.startswith('#') and line.strip() != '']

    with open(f"{dataset_path}/colmap_poses.txt", 'w') as out_file:
        out_file.write("IMAGE_ID QW QX QY QZ TX TY TZ IMAGE_NAME\n")

        # Process lines two by two (image metadata + points line)
        for i in range(0, len(lines), 2):
            meta_line = lines[i]

            parts = meta_line.split()

            image_id = parts[0]
            qw, qx, qy, qz = [f"{float(x):.6f}" for x in parts[1:5]]
            tx, ty, tz = [f"{float(x):.6f}" for x in parts[5:8]]
            image_name = parts[9]

            out_file.write(f"{image_id} {qw} {qx} {qy} {qz} {tx} {ty} {tz} {image_name}\n")

    # Delete COLAMP overhead that is not needed
    remove_overhead = f"rm -rf {dataset_path}/sparse {dataset_path}/sparse_txt {dataset_path}/database.db"
    result = subprocess.run(
        remove_overhead,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=True
    )

def compute_scale_factor(mocap_poses, colmap_poses):
    """
    Computes the scale factor between the metric MoCap poses and the unknown-scaled COLMAP poses.
    """
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

    # Keep only the norms furthest away
    num_keep = 10
    selected_indices = sorted_indices[-num_keep:]  # last half = farthest

    # Filter
    filtered_mocap_norms = mocap_norms[selected_indices]
    filtered_colmap_norms = colmap_norms[selected_indices]
    per_point_scales = filtered_mocap_norms / filtered_colmap_norms

    # Compute statistics
    scale = np.mean(per_point_scales)
    scale_std = np.std(per_point_scales)

    return scale, scale_std

def apply_scale_factor(mocap_poses, scale):
    """
    Applies a previously computed scale factor to a set of poses.
    mocap_poses has to be a dictionary with the corresponding image_id as key and a 4x4 homogenous matrix as value.
    """
    scaled_poses = {}
    for key, T in mocap_poses.items():
        T_scaled = T.copy()
        T_scaled[:3, 3] *= scale
        scaled_poses[key] = T_scaled
    return scaled_poses