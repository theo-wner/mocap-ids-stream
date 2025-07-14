"""
This script first initializes the camera and motion capture streams.
It then enters a loop where it captures frames from the camera and motion capture data.
Whenever a pose satisfies the "saving-conditions", it saves the current frame and pose to a file in a format suitable for COLMAP.

Author:
    Theodor Kapler <theodor.kapler@student.kit.edu>
"""

import os
import shutil
import cv2
from streams.ids_stream import IDSStream
from streams.mocap_stream import MoCapStream
import time

OUTPUT_DIR = "colmap_dataset"
IMAGES_DIR = os.path.join(OUTPUT_DIR, "images")
POSES_PATH = os.path.join(OUTPUT_DIR, "poses.txt")

def ensure_dirs():
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(IMAGES_DIR, exist_ok=True)

if __name__ == "__main__":
    ensure_dirs()
    # Initialize camera and motion capture streams
    cam_stream = IDSStream(frame_rate='max', 
                           exposure_time='auto', 
                           resize=None)
    mocap_stream = MoCapStream(client_ip="172.22.147.168", # 168 for workstation, 172 for laptop
                               server_ip="172.22.147.182", 
                               rigid_body_id=2, # 1 for calibration wand, 2 for camera rig
                               buffer_size=20)
    
    cam_stream.start_timing()
    mocap_stream.start_timing()

    time.sleep(1)  # Allow some time for streams to initialize

    with open(POSES_PATH, "w") as poses_file:
        poses_file.write("# image_name tx ty tz qx qy qz qw\n")
        print("Capturing colmap dataset. Press 'q' to quit.")
        img_idx = 0
        last_saved_pos = None  # Track last saved position

        while True:
            frame, info = cam_stream.getnext(return_tensor=False)
            if frame is not None:
                frame_vis = cv2.resize(frame, (1000, 1000))
                cv2.imshow("Camera", frame_vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Exiting...")
                break

            pos, rot, v_trans, v_rot = mocap_stream.get_interpolated_pose(
                info['timestamp'], marker_error_threshold=0.001, show_plot=False
            )

            if pos is not None and v_trans <= 0.5 and v_rot <= 0.5:
                should_save = False
                if last_saved_pos is None: # Don't save the first position
                    last_saved_pos = pos 
                else:
                    dist = ((pos[0] - last_saved_pos[0]) ** 2 +
                            (pos[1] - last_saved_pos[1]) ** 2 +
                            (pos[2] - last_saved_pos[2]) ** 2) ** 0.5
                    if dist >= 0.1:
                        should_save = True

                if should_save:
                    image_name = f"{img_idx:04d}.png"
                    cv2.imwrite(os.path.join(IMAGES_DIR, image_name), frame)
                    poses_file.write(
                        f"{image_name} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f} "
                        f"{rot[0]:.6f} {rot[1]:.6f} {rot[2]:.6f} {rot[3]:.6f}\n"
                    )
                    poses_file.flush()
                    print(f"Captured {image_name}")
                    os.system('aplay /usr/share/sounds/sound-icons/super-mario-coin-sound.wav > /dev/null 2>&1 &')
                    img_idx += 1
                    last_saved_pos = pos

    cam_stream.stop()
    mocap_stream.stop()
    cv2.destroyAllWindows()