"""
Script to check the consistency of time synchronization between the camera and motion capture system.

Author:
    Theodor Kapler <theodor.kapler@student.kit.edu>
"""

from data_streams.camera_stream import CameraStream
from data_streams.mocap_stream import MoCapStream
import time
import matplotlib.pyplot as plt

# Initialize streams
mocap_stream = MoCapStream(client_ip="172.22.147.172", server_ip="172.22.147.182")
camera_stream = CameraStream(frame_rate=30, exposure_time=20000, resize=(500, 500))
time.sleep(1)  # Allow streams to initialize

# Prepare for plotting
plt.ion()
fig, ax = plt.subplots()
lines = {
    "python": ax.plot([], [], label="Python Clock")[0],
    "camera": ax.plot([], [], label="Camera Timestamp")[0],
    "mocap": ax.plot([], [], label="MoCap Timestamp")[0],
}
ax.set_xlabel("Elapsed Time (s)")
ax.set_ylabel("Offset from ideal timeline (s)")
ax.set_title("Timeline Synchronization (Trend Removed)")
ax.legend()
ax.grid(True)
ax.set_ylim(-0.1, 0.1)  # Limit y-axis to ±0.1s

# Data containers
elapsed_times = []
python_offsets = []
camera_offsets = []
mocap_offsets = []

# Capture Loop
cnt = 0

try:
    while True:
        timestamp_python = time.time()
        _, timestamp_cam = camera_stream.get_current_frame()
        _, timestamp_mocap = mocap_stream.get_current_rigid_body_pose(rigid_body_id=1)

        if cnt == 0:
            first_timestamp_python = timestamp_python
            first_timestamp_cam = timestamp_cam
            first_timestamp_mocap = timestamp_mocap

        # Calculate relative time deltas in seconds
        elapsed = timestamp_python - first_timestamp_python
        delta_python = elapsed
        delta_camera = (timestamp_cam - first_timestamp_cam).total_seconds()
        delta_mocap = (timestamp_mocap - first_timestamp_mocap).total_seconds()

        # Subtract slope=1 trend (elapsed) to center timelines at zero
        python_offsets.append(delta_python - elapsed)
        camera_offsets.append(delta_camera - elapsed)
        mocap_offsets.append(delta_mocap - elapsed)
        elapsed_times.append(elapsed)

        # Update plot
        lines["python"].set_data(elapsed_times, python_offsets)
        lines["camera"].set_data(elapsed_times, camera_offsets)
        lines["mocap"].set_data(elapsed_times, mocap_offsets)
        ax.relim()
        ax.autoscale_view(scalex=True, scaley=False)  # Autoscale x, keep y fixed
        plt.pause(0.01)

        if elapsed > 10:
            print("Stopping after 10 seconds of data collection.")
            break

        cnt += 1
        time.sleep(0.1)

finally:
    mocap_stream.stop()
    camera_stream.stop()
    plt.ioff()
    plt.show()
