"""
Script to capture a frame from the camera and a pose from the MoCap system.

Author:
    Theodor Kapler <theodor.kapler@student.kit.edu>
"""

from data_streams.camera_stream import CameraStream
from data_streams.mocap_stream import MoCapStream
import cv2
from datetime import timedelta

# Initialize streams
mocap_stream = MoCapStream(client_ip="172.22.147.172", server_ip="172.22.147.182")
camera_stream = CameraStream(frame_rate=48, exposure_time=10000, resize=(500, 500))

cv2.namedWindow("Camera")

# Capture Loop
cnt = 0
print("Press 'c' to capture one frame and pose. Press 'q' to quit.")
try:
    while True:
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        if key == ord('c'):
            frame, timestamp_cam = camera_stream.get_current_frame()
            pose, timestamp_mocap = mocap_stream.get_current_rigid_body_pose(rigid_body_id=1)

            if cnt == 0:
                first_timestamp_cam = timestamp_cam
                first_timestamp_mocap = timestamp_mocap
                time_diff_cam = timedelta(seconds=0)
                time_diff_mocap = timedelta(seconds=0)
            else:
                time_diff_cam = timestamp_cam - first_timestamp_cam
                time_diff_mocap = timestamp_mocap - first_timestamp_mocap
                
            if pose:
                print(f"Pose: Position: {pose['position']}, Rotation: {pose['rotation']}")
                print(f'MoCap Timestamp: {time_diff_mocap}')
            else:
                print("No pose data received.")

            if frame is not None:
                cv2.imshow("Camera", frame)
                print(f'Camera Timestamp: {time_diff_cam}')
            else:
                print("No frame received.")

            cnt += 1

finally:
    mocap_stream.stop()
    camera_stream.stop()
    cv2.destroyAllWindows()
