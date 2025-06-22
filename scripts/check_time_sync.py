"""
Script to check the consistency of time synchronization between the camera and motion capture system.

Author:
    Theodor Kapler <theodor.kapler@student.kit.edu>
"""

from data_streams.cam_stream import CamStream
from data_streams.mocap_stream import MoCapStream
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Initialize streams
mocap_stream = MoCapStream(client_ip="172.22.147.172", server_ip="172.22.147.182", rigid_body_id=1)
cam_stream = CamStream(frame_rate=30, exposure_time=200, resize=(500, 500))
time.sleep(1)  # Allow streams to initialize

# Start time synchronization
mocap_stream.start_timing()
cam_stream.start_timing()
t0 = time.time()

# Prepare for plotting
plt.ion()
fig, ax = plt.subplots()
lines = {
    "cam": ax.plot([], [], label="Camera Timestamp")[0],
    "mocap": ax.plot([], [], label="MoCap Timestamp")[0],
}
ax.set_xlabel("Elapsed Time (s)")
ax.set_ylabel("Offset from ideal timeline (s)")
ax.legend()
ax.grid(True)

time.sleep(1)  # Allow some time for the streams to stabilize

elapsed_times = []
cam_offsets = []
mocap_offsets = []

last_timestamp_cam = None
last_timestamp_mocap = None

# Capture Loop
cnt = 0
try:
    while True:
        timestamp_python = time.time() - t0
        mocap_dict = mocap_stream.get_current_data()
        timestamp_mocap = mocap_dict['timestamp']
        cam_dict = cam_stream.get_current_data()
        timestamp_cam = cam_dict['timestamp']

        # Only update if both timestamps are new
        if (timestamp_cam != last_timestamp_cam and
            timestamp_mocap != last_timestamp_mocap):

            # Subtract slope=1 trend (delta_python) to center timelines at zero
            elapsed_times.append(timestamp_python)
            cam_offsets.append(timestamp_cam.total_seconds() - timestamp_python)
            mocap_offsets.append(timestamp_mocap.total_seconds() - timestamp_python)

            last_timestamp_cam = timestamp_cam
            last_timestamp_mocap = timestamp_mocap

            # Update plot
            lines["cam"].set_data(elapsed_times, cam_offsets)
            lines["mocap"].set_data(elapsed_times, mocap_offsets)
            ax.relim()
            ax.autoscale_view(scalex=True, scaley=True)  # Autoscale x, keep y fixed

            plt.pause(np.random.uniform(0.01, 0.1))  # Random pause to simulate real-time plotting

            if timestamp_python > 30:
                break

            cnt += 1

finally:
    mocap_stream.stop()
    cam_stream.stop()
    plt.ioff()
    plt.show()

    