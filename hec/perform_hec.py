import numpy as np
import cv2
from hec.colmap_utils import run_colmap, compute_scale_factor, apply_scale_factor, estimate_similarity_transform,convert_to_tum
from hec.general_utils import filter_poses, read_poses, compute_relative_motions

if __name__ == "__main__":
    #run_colmap(dataset_path = "./data/buddha")
    #filter_poses(dataset_path="./data/buddha")

    """
    mocap_poses = read_poses("./data/buddha/mocap_poses.txt")
    colmap_poses = read_poses("./data/buddha/colmap_poses.txt")

    scale, scale_std = compute_scale_factor(mocap_poses, colmap_poses)
    scaled_colmap_poses = apply_scale_factor(colmap_poses, scale)

    scale, R, t = estimate_similarity_transform(colmap_poses, mocap_poses)

    print("Estimated scale:", scale)

    R_A, t_A = compute_relative_motions(mocap_poses)
    R_B, t_B = compute_relative_motions(scaled_colmap_poses)

    R_X, t_X = cv2.calibrateHandEye(R_A, t_A, R_B, t_B, method=cv2.CALIB_HAND_EYE_TSAI)

    X = np.eye(4)
    X[:3, :3] = R_X
    X[:3, 3] = t_X.flatten()

    print("Hand-eye transform X (from mocap to camera):")
    print(X)
    """