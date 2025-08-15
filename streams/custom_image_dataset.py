"""
Representing a dataset suitable for the on-the-fly-nvs-implementation previously captured with the mocap-ids setup.

Author:
    Theodor Kapler <theodor.kapler@student.kit.edu>
"""
import numpy as np
import torch
import os
import cv2
from scipy.spatial.transform import Rotation as R

class CustomImageDatset():
    """
    A class representing a dataset previously captured with the mocap-ids setup.
    """
    def __init__(self, model_path):
        self.model_path = model_path
        self.images_dir = os.path.join(model_path, "images")
        self.current_index = 0  # Track current position for getnext()

        # Load intrinsics (only focal)
        with open(os.path.join(model_path, "intrinsics.txt"), "r") as intrinsics_file:
            first_line = intrinsics_file.readline()
            self.focal = float(intrinsics_file.readline().strip())
     
        # Load poses
        self.infos = {}
        with open(os.path.join(model_path, "poses.txt"), "r") as poses_file:
            first_line = poses_file.readline()
            for line in poses_file:
                line = line.strip().split(" ")
                image_id = int(line[0])
                qw = float(line[1])
                qx = float(line[2])
                qy = float(line[3])
                qz = float(line[4])
                tx = float(line[5])
                ty = float(line[6])
                tz = float(line[7])
                image_name = line[8]

                rotmat = R.from_quat([qx, qy, qz, qw]).as_matrix()
                Rt = np.eye(4)
                Rt[:3, :3] = rotmat
                Rt[:3, 3] = [tx, ty, tz]
                Rt = torch.from_numpy(Rt).float().cuda()
                focal = torch.from_numpy(np.array(self.focal)).float().unsqueeze(0).cuda()

                self.infos[image_id] = {
                    "Rt" : Rt,
                    "focal" : focal,
                    "image_name" : image_name,
                    "is_valid" : True, # always true because dataset has already been captured
                    "is_test" : False
                }
        
        # Create sorted list of image IDs for sequential access
        self.sorted_image_ids = sorted(self.infos.keys())

    def __len__(self):
        return len(self.infos)

    def get_image_size(self):
        image = cv2.imread(os.path.join(self.images_dir, "0000.png"))
        return image.shape[0], image.shape[1]
    
    def get_focal(self):
        return self.focal

    def getnext(self):
        # Get the current image ID
        image_id = self.sorted_image_ids[self.current_index]
        info = self.infos[image_id]
        
        # Load the image
        image_path = os.path.join(self.images_dir, info["image_name"])
        image = cv2.imread(image_path)
        image = torch.from_numpy(image).permute(2, 0, 1).cuda().float() / 255.0
        
        # Increment index for next call
        self.current_index += 1
        
        return image, info
