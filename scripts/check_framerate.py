"""
Module to quickly check the framerate of the CameraStream and MoCapStream classes.
Note: This is quick and dirty code for testing purposes only.

Author:
    Theodor Kapler <theodor.kapler@student.kit.edu>
"""

from data_streams.camera_stream import CameraStream
from data_streams.mocap_stream import MoCapStream
import time

def check_camera_framerate(camera_stream, duration=5):
    """
    Checks the rate in which the get_current_frame() method is able to return a new frame.

    Args:
        camera_stream (CameraStream): The CameraStream instance to check.
        duration (int): Duration in seconds for which to measure the framerate.

    Returns:
        float: The calculated framerate in frames per second (FPS).
    """

    time.sleep(2)
    print(f"Measuring CameraStream framerate for {duration} seconds...")
    
    start_time = time.time()
    frame_count = 0
    last_frame = None
    cnt = 0

    while time.time() - start_time < duration:
        frame = camera_stream.get_current_frame()
        if frame is not None and not (last_frame is frame):
            frame_count += 1
            last_frame = frame
        else:
            cnt += 1
    print(f"Skipped {cnt} identical frames.")

    elapsed = time.time() - start_time
    framerate = frame_count / elapsed
    return framerate

def check_mocap_framerate(mocap_stream, rigid_body_id=1, duration=5):
    """
    Checks the rate in which the get_current_rigid_body_pose() method is able to return a new pose.
    Args:
        mocap_stream (MoCapStream): The MoCapStream instance to check.
        rigid_body_id (int): The ID of the rigid body to check.
        duration (int): Duration in seconds for which to measure the framerate.

    Returns:
        float: The calculated framerate in poses per second (PPS).
    """

    time.sleep(2)
    print(f"Measuring MoCapStream framerate for {duration} seconds...")

    start_time = time.time()
    pose_count = 0
    last_pose = None

    cnt = 0
    while time.time() - start_time < duration:
        pose = mocap_stream.get_current_rigid_body_pose(rigid_body_id)
        if pose is not None and not last_pose is pose:
            pose_count += 1
            last_pose = pose
        else:
            cnt += 1

    print(f"Skipped {cnt} identical poses.")

    elapsed = time.time() - start_time
    framerate = pose_count / elapsed
    return framerate

if __name__ == "__main__":
    mocap_stream = MoCapStream(client_ip="172.22.147.172", server_ip="172.22.147.182")
    camera_stream = CameraStream(frame_rate=48, exposure_time=100, resize=(500, 500))
    time.sleep(2)

    try:
        camera_framerate = check_camera_framerate(camera_stream)
        mocap_framerate = check_mocap_framerate(mocap_stream)
        print(f"Camera framerate: {camera_framerate:.2f} FPS")
        print(f"MoCap framerate: {mocap_framerate:.2f} FPS")
    finally:
        mocap_stream.stop()
        camera_stream.stop()