import time
import cv2
from data_streams.cam_stream import CamStream
from data_streams.mocap_stream import MoCapStream
from data_streams.stream_matcher import StreamMatcher

if __name__ == "__main__":
    cam_stream = CamStream(frame_rate=30, exposure_time=200, resize=(500, 500))
    mocap_stream = MoCapStream(client_ip="172.22.147.172", server_ip="172.22.147.182", rigid_body_id=1)
    time.sleep(1)  # Allow streams to initialize
    cam_stream.start_timing()
    mocap_stream.start_timing()

    matcher = StreamMatcher(cam_stream, mocap_stream, maxlen=3000)
    matcher.start()
    time.sleep(1)  # Let buffers fill

    print("Press 'c' to capture and match, or 'q' to quit.")

    try:
        while True:
            cam_data = cam_stream.get_current_data()
            frame = cam_data.get('frame')
            if frame is not None:
                cv2.imshow("Camera", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('c'):
                print("Picture taken! Waiting for enough mocap poses after acquisition...")
                cam, mocap, dt = matcher.get_best_match(marker_error_threshold=0.02, require_tracking_valid=True)
                if cam and mocap:
                    print(f"Best match found! Δt={dt:.6f}s, error={mocap['mean_error']}, valid={mocap['tracking_valid']}")
                else:
                    print("No valid match found.")
            elif key == ord('q'):
                print("Exiting...")
                break
    finally:
        matcher.stop()
        cam_stream.stop()
        mocap_stream.stop()
        cv2.destroyAllWindows()