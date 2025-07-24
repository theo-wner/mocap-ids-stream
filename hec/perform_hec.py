import numpy as np
import cv2
from hec.general_utils import filter_poses, read_poses, compute_relative_motions
from hec.colmap_utils import convert_to_tum
from scipy.spatial.transform import Rotation as R
import itertools

def compute_hand_eye(R_A, t_A, R_B, t_B, method_name):
    method = {
        "horaud": cv2.CALIB_HAND_EYE_HORAUD,
        "tsai": cv2.CALIB_HAND_EYE_TSAI,
        "park": cv2.CALIB_HAND_EYE_PARK,
        "andreff": cv2.CALIB_HAND_EYE_ANDREFF,
        "daniilidis": cv2.CALIB_HAND_EYE_DANIILIDIS
    }[method_name.lower()]
    
    R_X, t_X = cv2.calibrateHandEye(R_A, t_A, R_B, t_B, method=method)
    X = np.eye(4)
    X[:3, :3] = R_X
    X[:3, 3] = t_X.flatten()
    return X

def compare_transforms(X_dict):
    print("\n=== Comparing Hand-Eye Transforms ===")
    keys = list(X_dict.keys())
    for i, j in itertools.combinations(keys, 2):
        X1 = X_dict[i]
        X2 = X_dict[j]

        X_rel = np.linalg.inv(X1) @ X2
        R_rel = X_rel[:3, :3]
        t_rel = X_rel[:3, 3]

        try:
            angle_rad = R.from_matrix(R_rel).magnitude()
            angle_deg = np.degrees(angle_rad)
        except ValueError:
            print(f"Between {i.upper()} and {j.upper()}: INVALID rotation matrix (skipped)\n")
            continue

        trans_diff = np.linalg.norm(t_rel)

        print(f"Between {i.upper()} and {j.upper()}:")
        print(f"  Rotation diff: {angle_deg:.4f} degrees")
        print(f"  Translation diff: {trans_diff:.6f} m\n")


if __name__ == "__main__":
    checkerboard_poses = read_poses("./data/hec_checkerboard/checkerboard_poses.txt")
    mocap_poses = read_poses("./data/hec_checkerboard/mocap_poses.txt")

    # Invert mocap poses to get Pose of robot base in tool
    for key in mocap_poses.keys():
        mocap_poses[key] = np.linalg.inv(mocap_poses[key])

    for key in checkerboard_poses.keys():
        checkerboard_poses[key] = np.linalg.inv(checkerboard_poses[key])

    """
    R_A, t_A, R_B, t_B = compute_relative_motions(checkerboard_poses, mocap_poses,
                                                  min_translation=0.2,
                                                  min_rotation_deg=30.0,
                                                  similarity_rot_thresh_deg=10.0,
                                                  similarity_trans_thresh=0.1)
    print(f"Found {len(R_A)} relative motions")
    """

    R_gripper2base, t_gripper2base, R_target2cam, t_target2cam = [], [], [], []

    for key in checkerboard_poses.keys():
        R_gripper2base.append(mocap_poses[key][:3, :3])
        t_gripper2base.append(mocap_poses[key][:3, 3])
        R_target2cam.append(checkerboard_poses[key][:3, :3])
        t_target2cam.append(checkerboard_poses[key][:3, 3])


    # Compute transforms using different methods
    methods = ["horaud", "tsai", "park", "andreff", "daniilidis"]
    X_dict = {}
    for method in methods:
        X = compute_hand_eye(R_target2cam, t_target2cam, R_gripper2base, t_gripper2base, method)
        X_dict[method] = X
        print(f"{method.upper()} hand-eye transform:\n{X}\n")

    # Compare all transforms
    compare_transforms(X_dict)

    # Use one of the transforms (e.g., Tsai) for pose transformation
    X = X_dict["horaud"]
    
    # Transform MoCap poses to checkerboard frame
    transformed_mocap_poses = {}
    for key, T_mocap in mocap_poses.items():
        T_transformed = np.linalg.inv(X @ T_mocap)
        transformed_mocap_poses[key] = T_transformed

    # Save transformed poses
    with open("./data/hec_checkerboard/transformed_mocap_poses.txt", "w") as f:
        for key, T in transformed_mocap_poses.items():
            f.write(f"{key} {' '.join(map(str, T.flatten()))}\n")


    convert_to_tum("./data/hec_checkerboard/checkerboard_poses.txt", "./data/hec_checkerboard/checkerboard_poses_tum.txt")
    convert_to_tum("./data/hec_checkerboard/transformed_mocap_poses.txt", "./data/hec_checkerboard/transformed_mocap_poses_tum.txt")
    convert_to_tum("./data/hec_checkerboard/mocap_poses.txt", "./data/hec_checkerboard/mocap_poses_tum.txt")
