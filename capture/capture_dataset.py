"""
This script provides functionality for capturing a dataset with camera and motion capture data.

Author:
    Theodor Kapler <theodor.kapler@student.kit.edu>
"""

import os
import cv2
from streams.ids_stream import IDSStream
from streams.mocap_stream import MoCapStream
import time

def capture_dataset(cam_stream, mocap_stream, output_dir, mode):
    """
    This function captures frames from the camera and motion capture data in a loop.
    Whenever a pose satisfies the "saving-conditions", it saves the current frame and pose to a file.
    """
    # Create Directory
    output_dir = get_unique_path(output_dir)
    images_dir = os.path.join(output_dir, "images")
    poses_path = os.path.join(output_dir, "poses.txt")
    os.makedirs(images_dir)

    # Set thresholds for auto capture mode
    min_dist = 0.1 # Threshold for the minimal allowed euclidean distance to the last captured image (m)
    max_v_trans = 0.01 # Threshold for the maximal allowed tranlational velocity (m/s)
    max_v_rot = 0.015 # Threshold for the maximal allowed rotational velocity (rad/s)

    with open(poses_path, "w") as poses_file:
        print("Capturing colmap dataset. Press 'q' to quit.")
        poses_file.write("IMAGE_ID QW QX QY QZ TX TY TZ IMAGE_NAME\n")
        img_idx = 0
        last_saved_pos = None  # Track last saved position

        while True:
            # Capture and display camera frame
            frame, info = cam_stream.getnext(return_tensor=False)
            if frame is not None:
                frame_vis = cv2.resize(frame, (1000, 1000))
                cv2.imshow("Camera", frame_vis)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Exiting...")
                break
            
            # Capture motion capture pose
            valid_pose, pos, rot, v_trans, v_rot = mocap_stream.get_interpolated_pose(
                info['timestamp'], marker_error_threshold=0.001, show_plot=False)
            
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
                    f"{img_idx} {rot[0]:.6f} {rot[1]:.6f} {rot[2]:.6f} {rot[3]:.6f} "
                    f"{pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f} {image_name}\n"
                )
                poses_file.flush()
                print(f"Captured {image_name}")
                os.system('aplay /usr/share/sounds/sound-icons/super-mario-coin-sound.wav > /dev/null 2>&1 &')
                img_idx += 1
                last_saved_pos = pos

    cv2.destroyAllWindows()

def get_unique_path(base_path):
    if not os.path.exists(base_path):
        return base_path
    counter = 1
    while True:
        new_path = f"{base_path}_{counter}"
        if not os.path.exists(new_path):
            return new_path
        counter += 1

if __name__ == "__main__":
    cam_stream = IDSStream(frame_rate='max', 
                           exposure_time='auto', 
                           white_balance='auto',
                           gain='auto',
                           gamma=1.0,
                           resize=None)
    
    mocap_stream = MoCapStream(client_ip="172.22.147.168", # 168 for workstation, 172 for laptop
                               server_ip="172.22.147.182", 
                               rigid_body_id=2, # 1 for calibration wand, 2 for camera rig
                               buffer_size=15)
    
    cam_stream.start_timing()
    mocap_stream.start_timing()
    time.sleep(1)

    capture_dataset(cam_stream, mocap_stream, output_dir='data/hec_colmap', mode='auto')

    cam_stream.stop()
    mocap_stream.stop()