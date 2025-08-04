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