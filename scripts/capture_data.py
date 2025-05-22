"""
Script to capture a frame from the camera and a pose from the MoCap system.

Author:
    Theodor Kapler <theodor.kapler@student.kit.edu>
"""

from data_streams.camera_stream import CameraStream
from data_streams.mocap_stream import MoCapStream
import cv2
import time

# Initialize streams
mocap_stream = MoCapStream(client_ip="172.22.147.172", server_ip="172.22.147.182")
camera_stream = CameraStream(frame_rate=48, exposure_time=10000, resize=(500, 500))

cv2.namedWindow("Camera")

# Capture Loop
try:
    print("Press 'c' to capture one frame and pose. Press 'q' to quit.")
    while True:
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        if key == ord('c'):
            start_time = time.time()
            frame = camera_stream.get_current_frame()
            pose = mocap_stream.get_current_rigid_body_pose(rigid_body_id=1)
            elapsed_time = (time.time() - start_time) * 1000000
            print(f"Time taken to get frame and pose: {elapsed_time:.4f} microseconds")

            if pose:
                print(f"Current pose: {pose}")
            else:
                print("No pose data received.")

            if frame is not None:
                cv2.imshow("Camera", frame)
            else:
                print("No frame received.")

finally:
    mocap_stream.stop()
    camera_stream.stop()
    cv2.destroyAllWindows()
