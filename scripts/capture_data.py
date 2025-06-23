import time
import cv2
from data_streams.cam_stream import CamStream
from data_streams.mocap_stream import MoCapStream

if __name__ == "__main__":
    # Initialize camera and motion capture streams
    cam_stream = CamStream(frame_rate=30, 
                           exposure_time=1000, 
                           resize=(500, 500))
    mocap_stream = MoCapStream(client_ip="172.22.147.172", 
                               server_ip="172.22.147.182", 
                               rigid_body_id=1, 
                               buffer_size=20)
    cam_stream.start()
    mocap_stream.start()
    time.sleep(1)
    cam_stream.start_timing()
    mocap_stream.start_timing()

    # Capture Loop
    print("Press 'c' to capture and match, or 'q' to quit.")
    while True:
        cam_data = cam_stream.get_current_data()
        frame = cam_data['frame']
        if frame is not None:
            cv2.imshow("Camera", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            print("Picture taken! Waiting for enough mocap poses after acquisition...")
            pos, rot = mocap_stream.get_interpolated_pose(cam_data, marker_error_threshold=0.001, show_plot=True)
        elif key == ord('q'):
            print("Exiting...")
            break

    # Close streams and windows   
    cam_stream.stop()
    mocap_stream.stop()
    cv2.destroyAllWindows()