"""
Script to check the consistency of time synchronization between the camera and motion capture system.

Author:
    Theodor Kapler <theodor.kapler@student.kit.edu>
"""

from data_streams.camera_stream import CameraStream
from data_streams.mocap_stream import MoCapStream
import time
import matplotlib.pyplot as plt
import numpy as np

# Initialize streams
mocap_stream = MoCapStream(client_ip="172.22.147.172", server_ip="172.22.147.182", rigid_body_id=1)
camera_stream = CameraStream(frame_rate=48, exposure_time=200, resize=(500, 500))
time.sleep(1)  # Allow streams to initialize

# Prepare for plotting
plt.ion()
fig, ax = plt.subplots()
lines = {
    "camera": ax.plot([], [], label="Camera Timestamp")[0],
    "mocap": ax.plot([], [], label="MoCap Timestamp")[0],
}
ax.set_xlabel("Elapsed Time (s)")
ax.set_ylabel("Offset from ideal timeline (s)")
ax.set_title("Timeline Synchronization (Trend Removed)")
ax.legend()
ax.grid(True)

elapsed_times = []
camera_offsets = []
mocap_offsets = []

# Add lists to store durations
durations_python = []
durations_mocap = []
durations_cam = []

# Capture Loop
cnt = 0
try:
    while True:
        # Measure time for each line
        t0 = time.time()
        timestamp_python = time.time()
        t1 = time.time()
        mocap_dict = mocap_stream.get_current_data()
        timestamp_mocap = mocap_dict['timestamp']
        t2 = time.time()
        cam_dict = camera_stream.get_current_data()
        timestamp_cam = cam_dict['timestamp']
        t3 = time.time()

        durations_python.append(t1 - t0)
        durations_mocap.append(t2 - t1)
        durations_cam.append(t3 - t2)

        if cnt == 0:
            first_timestamp_python = timestamp_python
            first_timestamp_cam = timestamp_cam
            first_timestamp_mocap = timestamp_mocap

        # Calculate relative time deltas in seconds
        delta_python = timestamp_python - first_timestamp_python
        delta_camera = (timestamp_cam - first_timestamp_cam).total_seconds()
        delta_mocap = (timestamp_mocap - first_timestamp_mocap).total_seconds()

        # Subtract slope=1 trend (delta_python) to center timelines at zero
        elapsed_times.append(delta_python)
        camera_offsets.append(delta_camera - delta_python)
        mocap_offsets.append(delta_mocap - delta_python)

        # Update plot
        lines["camera"].set_data(elapsed_times, camera_offsets)
        lines["mocap"].set_data(elapsed_times, mocap_offsets)
        ax.relim()
        ax.autoscale_view(scalex=True, scaley=True)  # Autoscale x, keep y fixed
        plt.pause(0.01)

        if delta_python > 30:
            print("Stopping after 10 seconds of data collection.")
            break

        cnt += 1

finally:
    print("Final mean timings:")
    print(f"Mean time for time.time(): {np.mean(durations_python)*1e3:.3f} ms")
    print(f"Mean time for mocap_stream.get_current_rigid_body_pose: {np.mean(durations_mocap)*1e3:.3f} ms")
    print(f"Mean time for camera_stream.get_current_frame: {np.mean(durations_cam)*1e3:.3f} ms")
    mocap_stream.stop()
    camera_stream.stop()
    plt.ioff()
    plt.show()
