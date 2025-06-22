"""
Script to visualize real-time motion capture data and camera feed.

Author:
    Theodor Kapler <theodor.kapler@student.kit.edu>
"""

from data_streams.cam_stream import CamStream
from data_streams.mocap_stream import MoCapStream
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R

# Initialize streams
mocap_stream = MoCapStream(client_ip="172.22.147.172", server_ip="172.22.147.182", rigid_body_id=1)
cam_stream = CamStream(frame_rate=30, exposure_time=20000, resize=(500, 500))

# Start the streams
mocap_stream.start()
cam_stream.start()

# Set up 3D plot
plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_zlim(0, 2)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Real-Time MoCap Pose (Coordinate Frame)')

quivers = {'x': None, 'y': None, 'z': None}

# Capture Loop
try:
    while True:
        cam_dict = cam_stream.get_current_data()
        frame = cam_dict['frame']

        mocap_dict = mocap_stream.get_current_data()
        pose = mocap_dict['rigid_body_pose']

        if pose:
            pos = np.array(pose['position'])           # [x, y, z]
            quat = np.array(pose['rotation'])          # [x, y, z, w]
            rot = R.from_quat(quat).as_matrix()

            # Axis vectors (scaled)
            x_axis = rot[:, 0] * 0.5
            y_axis = rot[:, 1] * 0.5
            z_axis = rot[:, 2] * 0.5

            # Remove old quivers
            for q in quivers.values():
                if q:
                    q.remove()

            # Draw new ones
            quivers['x'] = ax.quiver(*pos, *x_axis, color='r')
            quivers['y'] = ax.quiver(*pos, *y_axis, color='g')
            quivers['z'] = ax.quiver(*pos, *z_axis, color='b')

            fig.canvas.draw()
            fig.canvas.flush_events()

        if frame is not None:
            cv2.imshow("Camera", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    mocap_stream.stop()
    cam_stream.stop()
    cv2.destroyAllWindows()
    plt.ioff()
    plt.close()
