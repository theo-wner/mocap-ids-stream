"""
Module to quickly check the framerate of the IDSStream and MoCapStream classes.
Note: This is quick and dirty code for testing purposes only.

Author:
    Theodor Kapler <theodor.kapler@student.kit.edu>
"""

from streams.ids_stream import IDSStream
from streams.mocap_stream import MoCapStream
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
        mocap_dict = mocap_stream.getnext()
        timestamp = mocap_dict['timestamp']
        pose = mocap_dict['rigid_body'][mocap_stream.rigid_body_id]

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
        frame, info = cam_stream.getnext()
        timestamp = info['timestamp']

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
    """
    cam_stream = IDSStream(frame_rate='max', 
                           exposure_time='auto', 
                           white_balance='auto',
                           gain='auto',
                           gamma=1.5,
                           resize=(1000, 1000))
    """
    mocap_stream = MoCapStream(client_ip="172.22.147.168", # 168 for workstation, 172 for laptop
                               server_ip="172.22.147.182", 
                               rigid_body_id=2, # 1 for calibration wand, 2 for camera rig
                               buffer_size=15)

    try:
        #check_cam_framerate(cam_stream, duration=1)
        check_mocap_framerate(mocap_stream, duration=1)
    finally:
        mocap_stream.stop()
        #cam_stream.stop()