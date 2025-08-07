"""
This script first initializes the camera and motion capture streams.
It then enters a loop where it captures frames from the camera and motion capture data.
When the user presses 'c', it captures the current frame and retrieves best matching motion capture pose.

Author:
    Theodor Kapler <theodor.kapler@student.kit.edu>
"""

import cv2
from streams.ids_stream import IDSStream
from streams.mocap_stream import MoCapStream
from streams.stream_matcher import StreamMatcher
from scipy.spatial.transform import Rotation as R

if __name__ == "__main__":
    # Initialize camera and motion capture streams
    cam_stream = IDSStream(frame_rate='max', 
                           exposure_time='auto', 
                           white_balance='auto',
                           gain='auto',
                           gamma=1.5,
                           resize=None)
    
    mocap_stream = MoCapStream(client_ip="172.22.147.168", # 168 for workstation, 172 for laptop
                               server_ip="172.22.147.182", 
                               rigid_body_id=2, # 1 for calibration wand, 2 for camera rig
                               buffer_size=15)
    
    matcher = StreamMatcher(cam_stream, mocap_stream, 10, calib_dir='latest')
    matcher.start_timing()

    # Capture Loop
    cnt = 0
    while True:
        frame, info = matcher.getnext()
        if info['Rt'] == None:
            print(f"{cnt}th Rt None")
            cnt += 1

    # Close windows Streams 
    matcher.stop()
    cam_stream.stop()
    mocap_stream.stop()
    cv2.destroyAllWindows()