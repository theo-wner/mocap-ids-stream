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
import pandas as pd

# Initialize streams
mocap_stream = MoCapStream(client_ip="172.22.147.172", server_ip="172.22.147.182", rigid_body_id=1)
camera_stream = CameraStream(frame_rate=30, exposure_time=200, resize=(500, 500))
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

last_timestamp_cam = None
last_timestamp_mocap = None

# Capture Loop
cnt = 0
try:
    while True:
        timestamp_python = time.time()
        mocap_dict = mocap_stream.get_current_data()
        timestamp_mocap = mocap_dict['timestamp']
        cam_dict = camera_stream.get_current_data()
        timestamp_cam = cam_dict['timestamp']

        if cnt == 0:
            first_timestamp_python = timestamp_python
            first_timestamp_cam = timestamp_cam
            first_timestamp_mocap = timestamp_mocap

        if (timestamp_cam is not None and timestamp_mocap is not None and
            timestamp_cam != last_timestamp_cam and
            timestamp_mocap != last_timestamp_mocap):

            # Calculate relative time deltas in seconds
            delta_python = timestamp_python - first_timestamp_python
            delta_camera = (timestamp_cam - first_timestamp_cam).total_seconds()
            delta_mocap = (timestamp_mocap - first_timestamp_mocap).total_seconds()

            # Subtract slope=1 trend (delta_python) to center timelines at zero
            elapsed_times.append(delta_python)
            camera_offsets.append(delta_camera - delta_python)
            mocap_offsets.append(delta_mocap - delta_python)

            last_timestamp_cam = timestamp_cam
            last_timestamp_mocap = timestamp_mocap

            # Update plot
            lines["camera"].set_data(elapsed_times, camera_offsets)
            lines["mocap"].set_data(elapsed_times, mocap_offsets)
            ax.relim()
            ax.autoscale_view(scalex=True, scaley=True)  # Autoscale x, keep y fixed

            plt.pause(np.random.uniform(0.01, 0.1))  # Random pause to simulate real-time plotting

            if delta_python > 30:
                print("Stopping after 10 seconds of data collection.")
                break

            cnt += 1

finally:
    mocap_stream.stop()
    camera_stream.stop()
    plt.ioff()
    plt.show()

    