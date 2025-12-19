"""
This script lets the user perform a certain experiment to retrieve the latency difference between IDS Camera and MoCap data packets.
As a prerequisite, the camera has to have strong background lighting for the image processing pipline to work.
After calling the script, the user can start the experiment by pressing 'c' for capture on the keyboard.
Now an up-and-down movement of a OptiTrack tracking ball has to be performed in front of the camera, followed by pressing the button 'q' for quitting.
The timestamp of the tracking ball's highest point is automatically retrieved from the MoCap, as well as the camera data.
The user now sees a plot, which lets him keep the experiment run by pressing 'k' or reject it by pressing 'r'.

Author:
    Theodor Kapler <theodor.kapler@student.kit.edu>
"""

from streams.ids_stream import IDSStream
from streams.mocap_stream import MoCapStream
from streams.stream_matcher import StreamMatcher
import cv2
import time
import numpy as np

# Initialize streams
cam_stream = IDSStream(
    frame_rate=45,
    exposure_time='auto',
    white_balance='auto',
    gain='auto',
    gamma=1.0
)

mocap_stream = MoCapStream(
    client_ip="172.22.147.168",
    server_ip="172.22.147.182",
    buffer_size=1000
)

matcher = StreamMatcher(cam_stream, mocap_stream, rb_id=2)

times = []
time_diffs = []
start_time = time.time()

cv2.namedWindow("Camera + Latency", cv2.WINDOW_NORMAL)

plot_width = 800
plot_height = 1000
margin = 60

while True:
    frame, timestamp = cam_stream.getnext()
    matcher.getnext(return_tensor=False)
    if frame is None:
        continue

    cam_resized = cv2.resize(frame, (1000, plot_height))

    plot_canvas = np.ones((plot_height, plot_width, 3), dtype=np.uint8) * 255

    if times:
        t_min, t_max = min(times), max(times)
        d_min, d_max = min(time_diffs), max(time_diffs)
        t_range = t_max - t_min if t_max != t_min else 1
        d_range = d_max - d_min if d_max != d_min else 1

        cv2.line(plot_canvas, (margin, margin), (margin, plot_height - margin), (0, 0, 0), 2)  # Y axis
        cv2.line(plot_canvas, (margin, plot_height - margin), (plot_width - margin, plot_height - margin), (0, 0, 0), 2)  # X axis

        cv2.putText(plot_canvas, "Latency (ms)", (10, margin - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.putText(plot_canvas, "Time (s)", (plot_width - 120, plot_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        for t, d in zip(times, time_diffs):
            x = int(margin + (t - t_min) / t_range * (plot_width - 2*margin))
            y = int(plot_height - margin - (d - d_min) / d_range * (plot_height - 2*margin))
            cv2.circle(plot_canvas, (x, y), 6, (0, 0, 255), -1)

        for i in range(5):
            y_val = d_min + i * d_range / 4
            y_pos = int(plot_height - margin - i * (plot_height - 2*margin) / 4)
            cv2.line(plot_canvas, (margin-10, y_pos), (margin, y_pos), (0, 0, 0), 2)
            cv2.putText(plot_canvas, f"{y_val:.1f}", (2, y_pos+5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

        for i in range(5):
            x_val = t_min + i * t_range / 4
            x_pos = int(margin + i * (plot_width - 2*margin) / 4)
            cv2.line(plot_canvas, (x_pos, plot_height-margin), (x_pos, plot_height-margin+10), (0,0,0),2)
            cv2.putText(plot_canvas, f"{x_val:.1f}", (x_pos-15, plot_height-30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0),1)

    combined_width = cam_resized.shape[1] + plot_canvas.shape[1]
    combined_height = plot_height
    combined = np.ones((combined_height, combined_width, 3), dtype=np.uint8) * 255
    combined[0:plot_height, 0:cam_resized.shape[1]] = cam_resized
    combined[0:plot_height, cam_resized.shape[1]:] = plot_canvas

    cv2.imshow("Camera + Latency", combined)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):
        elapsed = time.time() - start_time
        latency = matcher.get_time_diff()
        if latency is not None:
            times.append(elapsed)
            time_diffs.append(latency)
    elif key == ord('q'):
        break

cam_stream.stop()
mocap_stream.stop()
cv2.destroyAllWindows()
