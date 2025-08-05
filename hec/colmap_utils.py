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
    # Extract translations
    mocap_t = np.array([mocap_poses[k][:3, 3] for k in mocap_poses.keys()])
    colmap_t = np.array([colmap_poses[k][:3, 3] for k in colmap_poses.keys()])

    # Center data
    mocap_centered = mocap_t - mocap_t.mean(axis=0)
    colmap_centered = colmap_t - colmap_t.mean(axis=0)

    # Compute norms (distances to center)
    mocap_norms = np.linalg.norm(mocap_centered, axis=1)
    colmap_norms = np.linalg.norm(colmap_centered, axis=1)

    # Compute scale per Point
    scales = mocap_norms / colmap_norms

    return np.mean(scales), np.std(scales)

def estimate_similarity_transform(mocap_poses, colmap_poses):
    """
    Estimate the similarity transform (scale, rotation, translation) 
    from source_points to target_points using the Umeyama method.
    """
    # Extract translations
    target_points = np.array([mocap_poses[k][:3, 3] for k in mocap_poses.keys()])
    source_points = np.array([colmap_poses[k][:3, 3] for k in colmap_poses.keys()])
    
    assert source_points.shape == target_points.shape

    mu_source = np.mean(source_points, axis=0)
    mu_target = np.mean(target_points, axis=0)

    src_centered = source_points - mu_source
    tgt_centered = target_points - mu_target

    # Covariance matrix
    cov_matrix = np.dot(tgt_centered.T, src_centered) / source_points.shape[0]

    # SVD
    U, D, Vt = np.linalg.svd(cov_matrix)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1

    R = np.dot(U, np.dot(S, Vt))
    var_src = np.var(src_centered, axis=0).sum()
    scale = np.trace(np.dot(np.diag(D), S)) / var_src

    t = mu_target - scale * np.dot(R, mu_source)

    # Transform the source (colmap) points
    source_transformed = scale * np.dot(R, source_points.T).T + t

    # Debug prints
    print("\nTarget (MoCap) points:\n", target_points)
    print("\nTransformed Source (COLMAP) points:\n", source_transformed)
    print("\nDifference (Target - Transformed):\n", target_points - source_transformed)

    return scale, R, t

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