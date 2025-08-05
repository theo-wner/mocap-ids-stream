import os
import numpy as np
from scipy.spatial.transform import Rotation as R

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
    Reads poses from a COLMAP-style pose file and returnes them as lists of rotation matrices and translation vectors.
    """
    rotations = []
    translations = []
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('#') or line.strip() == '' or line.startswith('IMAGE_ID'):
                continue
            parts = line.strip().split()
            qw, qx, qy, qz = map(float, parts[1:5])
            tx, ty, tz = map(float, parts[5:8])
            rot = R.from_quat([qx, qy, qz, qw]).as_matrix()
            trans = np.array([tx, ty, tz])
            rotations.append(rot)
            translations.append(trans)
    return rotations, translations

def convert_to_tum(input_path, output_path):
    """
    Converts a COLMAP-style pose file into a TUM-style Trajectory file for evalutaion with evo.
    """
    with open(input_path, 'r') as f:
        lines = f.readlines()

    tum_lines = []
    for idx, line in enumerate(lines):
        if line.strip().startswith('IMAGE_ID') or not line.strip():
            continue  # Skip header or empty lines

        tokens = line.strip().split()
        if len(tokens) < 8:
            continue  # Skip malformed lines

        image_id = int(tokens[0])
        qw, qx, qy, qz = map(float, tokens[1:5])
        tx, ty, tz = map(float, tokens[5:8])

        tum_line = f"{idx:.6f} {tx:.6f} {ty:.6f} {tz:.6f} {qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}"
        tum_lines.append(tum_line)

    with open(output_path, 'w') as f:
        f.write("\n".join(tum_lines))
        f.write("\n")

    print(f"Converted {len(tum_lines)} poses to TUM format in: {output_path}")