"""
MoCap Streaming Module for an OptiTrack System running on Motive
Contains wrapper class MoCapStream, that wraps the NatNetSDK Python Client

Author:
    Theodor Kapler <theodor.kapler@student.kit.edu>
"""

import numpy as np
from .NatNetSDK import NatNetClient
from datetime import timedelta

class MoCapStream:
    """
    A class to stream motion capture data from a NatNet server.

    Attributes: client - The NatNet client instance.
                rigid_body_poses - A dictionary to store the poses of all rigid bodies.
    """

    def __init__(self, client_ip, server_ip, rigid_body_id):
        """
        Initializes the MoCapStream class.

        Args:
            client_ip (str): The IP address of the NatNet client. 
            server_ip (str): The IP address of the Server (The PC running Motive)
            rigid_body_id (int): The ID of the rigid body to track.
        """
        # Initialize the NatNet client
        self.client = NatNetClient()
        self.client.set_client_address(client_ip)
        self.client.set_server_address(server_ip)
        self.client.set_use_multicast(False)
        
        # Set up listeners for rigid_bodies and frames (needed for time sync) --> No threading needed because the NatNet client handles this internally
        self.client.rigid_body_listener = self.rigid_body_listener
        self.client.new_frame_listener = self.new_frame_listener

        # Member variables to control the streaming
        self.rigid_body_id = rigid_body_id

        # Member variables to store the latest data
        self.timestamp = None
        self.rigid_body_pose = None

        # Check if the client is connected
        if not self.client.run("d"):
            raise RuntimeError("Failed to start NatNet client.")
        
        # Supress printing of the NatNet client
        self.client.set_print_level(0)  # print_level = 0 off, print_level = 1 on, print_level = >1 on / print every nth mocap frame:
                                        # Addionally, comment out line 1663 in NatNetClient.py

    def new_frame_listener(self, frame_data):
        """
        Listener for new frame data to capture the latest timestamp.

        Args:
            frame_data (dict): The data of the new frame, which includes a timestamp.
        """
        timestamp = frame_data.get('timestamp')
        self.timestamp = timedelta(seconds=timestamp)

    def rigid_body_listener(self, rigid_body_id, position, rotation):
        """
        Listener to handle the rigid body pose data.

        Args:
            id (int): The ID of the rigid body.
            position (tuple): The position of the rigid body (x, y, z).
            rotation (tuple): The rotation of the rigid body (qx, qy, qz, qw).
        """
        if rigid_body_id == self.rigid_body_id:
            self.rigid_body_pose = {
                'position': np.array(position),
                'rotation': np.array(rotation),
            }

    def get_current_data(self):
        """
        Retrieves the pose of a certain rigid body.

        Args:
            id (int): The ID of the rigid body.

        Returns:
            dict: A dictionary containing the timestamp and the pose of the rigid body.
        """
        return {'timestamp': self.timestamp,
                'rigid_body_pose': self.rigid_body_pose}
    
    def stop(self):
        """
        Stops the NatNet client and releases resources.
        """
        self.client.shutdown()


