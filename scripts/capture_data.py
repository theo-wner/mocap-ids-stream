"""
Script to capture a frame from the camera and a pose from the MoCap system.

Author:
    Theodor Kapler <theodor.kapler@student.kit.edu>
"""

from data_streams.cam_stream import CamStream
from data_streams.mocap_stream import MoCapStream
import cv2
import time

# Initialize streams
mocap_stream = MoCapStream(client_ip="172.22.147.172", 
                           server_ip="172.22.147.182", 
                           rigid_body_id=1, 
                           buffer_size=300)
cam_stream = CamStream(frame_rate=30, 
                       exposure_time=10000, 
                       resize=(500, 500))

# Start the streams
mocap_stream.start()
cam_stream.start()
time.sleep(1)  # Allow some time for the streams to initialize

# Start time synchronization
mocap_stream.start_timing()
cam_stream.start_timing()

cv2.namedWindow("Camera")

# Capture Loop
print("Press 'c' to capture one frame and pose. Press 'q' to quit.")
try:
    while True:
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        if key == ord('c'):
            cam_dict = cam_stream.get_current_data()
            timestamp_cam = cam_dict['timestamp']
            frame = cam_dict['frame']

            mocap_dict = mocap_stream.get_current_data()
            timestamp_mocap = mocap_dict['timestamp']
            pose = mocap_dict['rigid_body_pose']
                
            if pose:
                print(f"Pose: Position: {pose['position']}, Rotation: {pose['rotation']}")
                print(f'MoCap Timestamp: {timestamp_mocap}')
            else:
                print("No pose data received.")

            if frame is not None:
                cv2.imshow("Camera", frame)
                print(f'Camera Timestamp: {timestamp_cam}')
            else:
                print("No frame received.")

finally:
    mocap_stream.stop()
    cam_stream.stop()
    cv2.destroyAllWindows()
