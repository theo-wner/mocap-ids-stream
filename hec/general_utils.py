import os
import numpy as np
from scipy.spatial.transform import Rotation as R
import itertools

def filter_poses(dataset_path):
    """
    Deletes MoCap poses that have no corresponding image in /images
    """
    mocap_path = os.path.join(dataset_path, "mocap_poses.txt")
    images_dir = os.path.join(dataset_path, "images")
    existing_images = set(os.listdir(images_dir))

    with open(mocap_path, 'r') as f:
        lines = f.readlines()

    with open(mocap_path, 'w') as out_f:
        for line in lines:
            if line.startswith('#') or line.strip() == '' or line.startswith('IMAGE_ID'):
                out_f.write(line)
                continue

            parts = line.strip().split()

            image_name = parts[8]
            if image_name in existing_images:
                out_f.write(line)

def read_poses(filename):
    """
    Reads poses from a COLMAP-style pose file and saves them to a dictionary with the corresponding image_id as key and a 4x4 homogenous matrix as value.
    """
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

def rotation_difference_deg(R1, R2):
    """Returns angular difference between two rotation matrices in degrees."""
    R_diff = R.from_matrix(R1.T @ R2)
    return R_diff.magnitude() * 180 / np.pi

def translation_difference(t1, t2):
    """Euclidean distance between translation vectors."""
    return np.linalg.norm(t1 - t2)

def compute_relative_motions(checkerboard_poses, mocap_poses, 
                              min_translation=0.01, 
                              min_rotation_deg=2.0,
                              similarity_rot_thresh_deg=1.0,
                              similarity_trans_thresh=0.01):
    """
    Computes diverse relative motions for hand-eye calibration from eye and hand poses.

    Filters out motions that are too small or too similar to existing ones.
    """
    common_ids = sorted(set(checkerboard_poses.keys()) & set(mocap_poses.keys()))
    R_A, t_A, R_B, t_B = [], [], [], []

    for i, j in itertools.combinations(common_ids, 2):
        T_Ai = mocap_poses[i]
        T_Aj = mocap_poses[j]
        T_Bi = checkerboard_poses[i]
        T_Bj = checkerboard_poses[j]

        # Relative motions
        T_A_rel = np.linalg.inv(T_Ai) @ T_Aj
        T_B_rel = np.linalg.inv(T_Bi) @ T_Bj

        R_Aij = T_A_rel[:3, :3]
        t_Aij = T_A_rel[:3, 3]
        R_Bij = T_B_rel[:3, :3]
        t_Bij = T_B_rel[:3, 3]

        # Check if motion is large enough
        rot_deg = R.from_matrix(R_Aij).magnitude() * 180 / np.pi
        trans_norm = np.linalg.norm(t_Aij)

        if rot_deg < min_rotation_deg or trans_norm < min_translation:
            continue

        # Check if motion is sufficiently different from others
        is_similar = False
        for Ra_existing, ta_existing in zip(R_A, t_A):
            rot_diff = rotation_difference_deg(Ra_existing, R_Aij)
            trans_diff = translation_difference(ta_existing, t_Aij)
            if rot_diff < similarity_rot_thresh_deg and trans_diff < similarity_trans_thresh:
                is_similar = True
                break

        if not is_similar:
            R_A.append(R_Aij)
            t_A.append(t_Aij)
            R_B.append(R_Bij)
            t_B.append(t_Bij)

    return R_A, t_A, R_B, t_B