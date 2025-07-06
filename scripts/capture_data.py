"""
This script first initializes the camera and motion capture streams.
It then enters a loop where it captures frames from the camera and motion capture data.
When the user presses 'c', it captures the current frame and retrieves best matching motion capture pose.

Author:
    Theodor Kapler <theodor.kapler@student.kit.edu>
"""

import time
import cv2
from data_streams.cam_stream import CamStream
from data_streams.mocap_stream import MoCapStream

if __name__ == "__main__":
    # Initialize camera and motion capture streams
    cam_stream = CamStream(frame_rate=30, 
                           exposure_time=20000, 
                           resize=None)
    mocap_stream = MoCapStream(client_ip="172.22.147.168", # 168 for workstation, 172 for laptop
                               server_ip="172.22.147.182", 
                               rigid_body_id=2, # 1 for calibration wand, 2 for camera rig
                               buffer_size=20)
    cam_stream.start()
    mocap_stream.start()
    time.sleep(1)
    cam_stream.start_timing()
    mocap_stream.start_timing()
    time.sleep(1)

    # Capture Loop
    print("Press 'c' to capture and match, or 'q' to quit.")
    while True:
        cam_data = cam_stream.get_current_data()
        frame = cam_data['frame']
        if frame is not None:
            frame_vis = cv2.resize(frame, (1000, 1000))
            cv2.imshow("Camera", frame_vis)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            print("Picture taken! Waiting for enough mocap poses after acquisition...")
            pos, rot, v_trans, v_rot = mocap_stream.get_interpolated_pose(cam_data, marker_error_threshold=0.001, show_plot=False)
            if pos is not None:
                print(f"Linear velocity: {v_trans:.2f} m/s, Angular velocity: {v_rot:.2f} rad/s")
        elif key == ord('q'):
            print("Exiting...")
            break

    # Close streams and windows   
    cam_stream.stop()
    mocap_stream.stop()
    cv2.destroyAllWindows()