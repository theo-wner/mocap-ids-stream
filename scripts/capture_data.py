"""
Script to capture a frame from the camera and a pose from the MoCap system.

Author:
    Theodor Kapler <theodor.kapler@student.kit.edu>
"""

from data_streams.camera_stream import CameraStream
from data_streams.mocap_stream import MoCapStream
import cv2

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
            frame = camera_stream.get_current_frame()
            pose, timestamp = mocap_stream.get_current_rigid_body_pose(rigid_body_id=1)

            if pose:
                print(f"Pose for timestamp {timestamp}: Position: {pose['position']}, Rotation: {pose['rotation']}")
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
