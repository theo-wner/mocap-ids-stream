"""
This script provides functionality for capturing a dataset with camera and motion capture data.

Author:
    Theodor Kapler <theodor.kapler@student.kit.edu>
"""

import os
import cv2

def capture_dataset(stream_matcher, dataset_path, mode):
    """
    This function captures frames from the camera and motion capture data in a loop.
    Whenever a pose satisfies the "saving-conditions", it saves the current frame and pose to a file.

    Args:
        stream_matcher (StreamMatcher): The stream matcher object that synchronizes camera and motion capture data.
        dataset_path (str): The path where the dataset will be saved.
        mode (str): The mode of capturing, either 'auto' or 'manually'.
    """
    # Create Directories 
    dataset_path = get_unique_path(dataset_path)

    images_dir = os.path.join(dataset_path, "images")
    os.makedirs(images_dir)

    poses_dir = os.path.join(dataset_path, "sparse", "0")
    os.makedirs(poses_dir)
    poses_path = os.path.join(poses_dir, "images_mocap.txt")

    # Set thresholds for auto capture mode
    min_dist = 0.1 # Threshold for the minimal allowed euclidean distance to the last captured image (m)
    max_v_trans = 0.01 # Threshold for the maximal allowed tranlational velocity (m/s)
    max_v_rot = 0.015 # Threshold for the maximal allowed rotational velocity (rad/s)

    with open(poses_path, "w") as poses_file:
        print("Capturing colmap dataset. Press 'q' to quit.")
        poses_file.write("# Image list with two lines of data per image:\n")
        poses_file.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        poses_file.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        poses_file.write("# Number of images: PLACEHOLDER, mean observations per image: 0\n")
        poses_file.write("# These poses have been captured with a MoCap system, therefore CAMERA_ID is set to 0\n")
        img_idx = 0
        last_saved_pos = None  # Track last saved position

        while True:
            # Get Stream data
            frame, info = stream_matcher.getnext(return_tensor=False, show_plot=False)
            valid_pose = info['is_valid']

            if not valid_pose:
                continue

            pos = info['pose']['pos']
            rot = info['pose']['rot']
            v_trans = info['pose_velocity']['pos']
            v_rot = info['pose_velocity']['rot']

            if frame is not None:
                frame_vis = cv2.resize(frame, (1000, 1000))
                cv2.imshow("Camera", frame_vis)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print("Exiting...")
                break
            
            if not valid_pose:
                continue

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
                if v_trans <= max_v_trans and v_rot <= max_v_rot:
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
                    f"{pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f} 0 {image_name}\n\n"
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

def get_unique_path(base_path):
    if not os.path.exists(base_path):
        return base_path
    counter = 1
    while True:
        new_path = f"{base_path}_{counter}"
        if not os.path.exists(new_path):
            return new_path
        counter += 1