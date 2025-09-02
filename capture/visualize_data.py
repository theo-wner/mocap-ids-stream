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
                            gamma=1.0)
    
    mocap_stream = MoCapStream(client_ip="172.22.147.168", # 168 for workstation, 172 for laptop
                               server_ip="172.22.147.182", 
                               rigid_body_id=2, # 1 for calibration wand, 2 for camera rig
                               buffer_size=15)
    
    matcher = StreamMatcher(cam_stream, mocap_stream, 10, calib_path=None, downsampling=None)
    matcher.start_timing()

    # Capture Loop
    print("Press 'c' to capture and match, or 'q' to quit.")
    while True:
        frame, info = cam_stream.getnext()
        if frame is not None:
            frame = cv2.resize(frame, (1000, 1000))
            cv2.imshow("Camera", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            print("Picture taken! Waiting for enough mocap poses after acquisition...")
            frame, info = matcher.getnext(for_image=(frame, info), show_plot=True)
            if info['is_valid']:
                print(info)
            else:
                print("No valid pose")
        elif key == ord('q'):
            print("Exiting...")
            break

    # Close windows Streams 
    matcher.stop()
    cam_stream.stop()
    mocap_stream.stop()
    cv2.destroyAllWindows()