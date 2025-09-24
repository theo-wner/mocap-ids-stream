"""
This script provides functionality for capturing a dataset with camera and motion capture data.

Author:
    Theodor Kapler <theodor.kapler@student.kit.edu>
"""

import os
import cv2
import numpy as np

def capture_dataset(stream_matcher, dataset_path, mode, downsampling=1, undistort=False):
    """
    This function captures frames from the camera and motion capture data in a loop.
    Whenever a pose satisfies the "saving-conditions", it saves the current frame and pose to a file.

    Args:
        stream_matcher (StreamMatcher): The stream matcher object that synchronizes camera and motion capture data.
        dataset_path (str): The path where the dataset will be saved.
        mode (str): The mode of capturing, either 'auto' or 'manually'.
    """
    # Create Directories 
    images_dir = os.path.join(dataset_path, "images")
    os.makedirs(images_dir)
    data_dir = os.path.join(dataset_path, "sparse", "0")
    os.makedirs(data_dir)
    if stream_matcher.get_calib_path() == None:
        poses_path = os.path.join(dataset_path, "sparse", "0", "images_mocap.txt") # Dir structure in case of capturing a calibration dataset
    else:
        poses_path = os.path.join(dataset_path, "sparse", "0", "images.txt") # Dir structure in case of capturing a normal dataset

    # Set thresholds for auto capture mode
    min_dist = 0.05 # Threshold for the minimal allowed euclidean distance to the last captured image (m)
    max_m_pos = 0.5 # Threshold for the maximal allowed positional movement (std of the pose buffer)
    max_m_rot = 0.05 # Threshold for the maximal allowed rotational movement (std of the pose buffer)

    with open(poses_path, "w") as poses_file:
        print("Capturing dataset. Press 'q' to quit.")
        poses_file.write("# Image list with two lines of data per image:\n")
        poses_file.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME, MEAN_MARKER_ERROR, TIME_DIFF\n")
        poses_file.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        poses_file.write("# Number of images: PLACEHOLDER, mean observations per image: 0\n")
        poses_file.write("# These poses have been captured with a MoCap system\n")
        img_idx = 0
        last_saved_pos = None  # Track last saved position

        while True:
            # Get Stream data
            frame, info = stream_matcher.getnext(return_tensor=False)

            if info == None:
                continue

            if downsampling != 1:
                frame = cv2.resize(frame, (frame.shape[1] // downsampling, frame.shape[0] // downsampling), interpolation=cv2.INTER_AREA)

            if undistort == True:
                distortion = info["distortion"]
                camera_matrix = info["camera_matrix"]
                fx = camera_matrix[0][0] / downsampling
                fy = camera_matrix[1][1] / downsampling
                h, w = stream_matcher.get_image_size()
                cx = (w / 2) / downsampling
                cy = (h / 2) / downsampling
                new_camera_matrix = np.array([[fx, 0, cx],
                                              [0, fy, cy],
                                              [0, 0, 1]])
                frame = cv2.undistort(frame, camera_matrix, distortion, None, new_camera_matrix)

            pos = info["pos"]
            rot = info["rot"]
            m_pos = info["m_pos"]
            m_rot = info["m_rot"]
            mean_error = info["mean_error"]
            time_diff = info["time_diff"] * 1000

            if frame is not None:
                frame_vis = cv2.resize(frame, (1000, 1000))
                cv2.imshow("Camera", frame_vis)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print("Exiting...")
                break

            should_save = False

            if mode == 'manually':
                if key == ord('c'):
                    should_save = True
            
            elif mode == 'auto':
                # Skip first pose
                if last_saved_pos is None:
                    last_saved_pos = pos
                    continue
                # Check if the pose meets the saving conditions
                if m_pos <= max_m_pos and m_rot <= max_m_rot:
                    dist = ((pos[0] - last_saved_pos[0]) ** 2 +
                            (pos[1] - last_saved_pos[1]) ** 2 +
                            (pos[2] - last_saved_pos[2]) ** 2) ** 0.5
                    if dist >= min_dist:
                        should_save = True

            else:
                raise ValueError("Possible Values for 'mode' are 'auto' or 'manually'")

            # Save the frame and pose if conditions are met
            if should_save:
                image_name = f"{img_idx:04d}.png"
                cv2.imwrite(os.path.join(images_dir, image_name), frame)
                poses_file.write(
                    f"{img_idx} {rot[3]:.6f} {rot[0]:.6f} {rot[1]:.6f} {rot[2]:.6f} "
                    f"{pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f} 1 {image_name} {mean_error:.6f} {time_diff:.6f}\n\n"
                )
                poses_file.flush()
                print(f"Captured {image_name}")
                os.system('aplay /usr/share/sounds/sound-icons/super-mario-coin-sound.wav > /dev/null 2>&1 &')
                img_idx += 1
                last_saved_pos = pos

    cv2.destroyAllWindows()
    
    # Update the header with the correct number of images
    with open(poses_path, "r") as f:
        content = f.read()

    updated_content = content.replace("PLACEHOLDER", f"{img_idx}")
    
    with open(poses_path, "w") as f:
        f.write(updated_content)