import time
import numpy as np
import matplotlib.pyplot as plt
from data_streams.camera_stream import CameraStream

def main():
    camera_stream = CameraStream(frame_rate=30, exposure_time=200, resize=(500, 500))
    time.sleep(1)  # Allow camera to initialize

    elapsed_times = []
    camera_timestamps = []

    last_camera_timestamp = None
    t0 = time.time()

    print("Collecting camera timestamps. Press Ctrl+C to stop...")

    try:
        while True:
            cam_dict = camera_stream.get_current_data()
            camera_timestamp = cam_dict['timestamp']
            if camera_timestamp is not None and camera_timestamp != last_camera_timestamp:
                elapsed = time.time() - t0
                elapsed_times.append(elapsed)
                camera_timestamps.append(camera_timestamp)
                last_camera_timestamp = camera_timestamp
                print(f"Elapsed: {elapsed:.3f}s, Camera Timestamp: {camera_timestamp}")
            time.sleep(0.001)  # Prevent busy-waiting
    except KeyboardInterrupt:
        print("Data collection stopped.")

    # Plot results
    plt.figure()
    plt.plot(elapsed_times, [ts.total_seconds() if hasattr(ts, "total_seconds") else ts for ts in camera_timestamps], label="Camera Timestamps")
    plt.xlabel("Elapsed Time (s)")
    plt.ylabel("Camera Timestamp")
    plt.title("Camera Timestamps Over Time")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Assuming elapsed_times and camera_timestamps_sec are your lists
    camera_timestamps_sec = [ts.total_seconds() for ts in camera_timestamps]

    # Fit a linear trend
    x = np.array(elapsed_times)
    y = np.array(camera_timestamps_sec)
    coeffs = np.polyfit(x, y, 1)
    trend = np.polyval(coeffs, x)

    # Detrend
    detrended = y - trend

    # Plot detrended data
    plt.figure()
    plt.plot(x, detrended, label="Detrended Camera Timestamps")
    plt.xlabel("Elapsed Time (s)")
    plt.ylabel("Offset from ideal timeline (s)")
    plt.title("Detrended Camera Timestamps")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()