"""
Module to quickly check the framerate of the CamStream and MoCapStream classes.
Note: This is quick and dirty code for testing purposes only.

Author:
    Theodor Kapler <theodor.kapler@student.kit.edu>
"""

from data_streams.cam_stream import CamStream
from data_streams.mocap_stream import MoCapStream
import time

def check_mocap_framerate(mocap_stream, duration=5):
    """
    Measures the framerate of the MoCapStream by counting the number of unique poses and timestamps for a specific rigid body.

    Args:
        mocap_stream (MoCapStream): The MoCapStream instance to check.
        rigid_body_id (int): The ID of the rigid body to check.
        duration (int): Duration in seconds for which to measure the framerate.

    Returns:
        float: The calculated framerate in poses per second (PPS).
    """

    print("-------------------------------------------------------------------")
    print(f"Measuring MoCapStream framerate for {duration} seconds...")

    start_time = time.time()
    different_timestamps = 0
    equal_timestamps = 0
    last_timestamp = None

    different_poses = 0
    equal_poses = 0
    last_pose = None

    while time.time() - start_time < duration:
        mocap_dict = mocap_stream.get_current_data()
        timestamp = mocap_dict['timestamp']
        pose = mocap_dict['rigid_body_pose']

        if pose is not None and not (last_pose is pose):
            different_poses += 1
            last_pose = pose
        else:
            equal_poses += 1
        if timestamp is not None and not last_timestamp == timestamp:
            different_timestamps += 1
            last_timestamp = timestamp
        else:
            equal_timestamps += 1

    elapsed = time.time() - start_time
    timestamps_per_second = different_timestamps / elapsed
    poses_per_second = different_poses / elapsed
    
    print(f"Skipped {equal_timestamps} identical timestamps.")
    print(f"Skipped {equal_poses} identical poses.")
    print(f"MoCap is running at {timestamps_per_second:.2f} Timestamps Per Second (TPS)")
    print(f"MoCap is running at {poses_per_second:.2f} Poses Per Second (PPS)")
    print("-------------------------------------------------------------------")


def check_cam_framerate(cam_stream, duration=5):
    """
    Measures the framerate of the CamStream by counting the number of unique frames and timestamps.

    Args:
        cam_stream (CamStream): The CamStream instance to check.
        duration (int): Duration in seconds for which to measure the framerate.

    Returns:
        float: The calculated framerate in frames per second (FPS).
    """

    print("-------------------------------------------------------------------")
    print(f"Measuring CamStream framerate for {duration} seconds...")
    
    start_time = time.time()
    different_timestamps = 0
    equal_timestamps = 0
    last_timestamp = None

    different_frames = 0
    equal_frames = 0
    last_frame = None

    while time.time() - start_time < duration:
        cam_dict = cam_stream.get_current_data()
        timestamp = cam_dict['timestamp']
        frame = cam_dict['frame']

        if timestamp is not None and not last_timestamp == timestamp:
            different_timestamps += 1
            last_timestamp = timestamp
        else:
            equal_timestamps += 1
        if frame is not None and not (last_frame is frame):
            different_frames += 1
            last_frame = frame
        else:
            equal_frames += 1

    elapsed = time.time() - start_time
    timestamps_per_second = different_timestamps / elapsed
    frames_per_second = different_frames / elapsed

    print(f"Skipped {equal_timestamps} identical timestamps.")
    print(f"Skipped {equal_frames} identical frames.")
    print(f"Camera is running at {timestamps_per_second:.2f} Timestamps Per Second (TPS)")
    print(f"Camera is running at {frames_per_second:.2f} Frames Per Second (FPS)")
    print("-------------------------------------------------------------------")

if __name__ == "__main__":
    mocap_stream = MoCapStream(client_ip="172.22.147.172", server_ip="172.22.147.182", rigid_body_id=1, buffer_size=300)
    cam_stream = CamStream(frame_rate=30, exposure_time=100, resize=(500, 500))

    mocap_stream.start()
    cam_stream.start()
    time.sleep(1)

    try:
        check_cam_framerate(cam_stream, duration=1)
        check_mocap_framerate(mocap_stream, duration=1)
    finally:
        mocap_stream.stop()
        cam_stream.stop()