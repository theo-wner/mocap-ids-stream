import time
import cv2
from data_streams.cam_stream import CamStream
from data_streams.mocap_stream import MoCapStream
from data_streams.stream_matcher import StreamMatcher

if __name__ == "__main__":
    cam_stream = CamStream(frame_rate=30, exposure_time=200, resize=(500, 500))
    mocap_stream = MoCapStream(client_ip="172.22.147.172", server_ip="172.22.147.182", rigid_body_id=1, buffer_size=300)

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
                mocap, dt = mocap_stream.get_best_match(query_cam_data=cam_data)
                if mocap:
                    print(f"Best match found! Δt={dt:.6f}s, error={mocap['mean_error']}, valid={mocap['tracking_valid']}")
                else:
                    print("No valid match found.")
            elif key == ord('q'):
                print("Exiting...")
                break
    finally:
        cam_stream.stop()
        mocap_stream.stop()
        cv2.destroyAllWindows()