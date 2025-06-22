import time
import cv2
from data_streams.cam_stream import CamStream
from data_streams.mocap_stream import MoCapStream

if __name__ == "__main__":
    cam_stream = CamStream(frame_rate=30, 
                           exposure_time=10000, 
                           resize=(500, 500))
    mocap_stream = MoCapStream(client_ip="172.22.147.172", 
                               server_ip="172.22.147.182", 
                               rigid_body_id=1, 
                               buffer_size=20)

    cam_stream.start()
    mocap_stream.start()
    time.sleep(1)  # Allow streams to initialize
    
    cam_stream.start_timing()
    mocap_stream.start_timing()

    print("Press 'c' to capture and match, or 'q' to quit.")

    try:
        while True:
            cam_data = cam_stream.get_current_data()
            frame = cam_data['frame']
            if frame is not None:
                cv2.imshow("Camera", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('c'):
                print("Picture taken! Waiting for enough mocap poses after acquisition...")
                mocap, dt, buffer_idx = mocap_stream.get_best_match(query_cam_data=cam_data)
                if mocap:
                    dt = dt * 1000  # Convert to milliseconds
                    mean_error = mocap['mean_error'] * 1000  # Convert to millimeters
                    print(f"Best match found! Δt={dt:.2f}ms, error={mean_error:.2f}mm, valid={mocap['tracking_valid']}, buffer_index={buffer_idx}")
                else:
                    print("No valid match found.")
            elif key == ord('q'):
                print("Exiting...")
                break
    finally:
        cam_stream.stop()
        mocap_stream.stop()
        cv2.destroyAllWindows()