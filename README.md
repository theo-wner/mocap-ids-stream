# MoCap-IDS-Stream - Documentation

This repository provides a framework for capturing and processing camera and motion capture data using the IDS U3-31J0CP camera and the OptiTrack Motive software. 
It includes the classes `IDSStream` and `MoCapStream` for handling IDS camera and OptiTrack motion capture data streams.
Also, it contains scripts for capture visualization, capturing datasets suitable for COLMAP, checking time synchronization, and verifying frame rates.

## Prerequisites
Both the IDSStream and MoCapStream classes are client classes that require specific SDKs to be installed on your system.
For the IDS Camera, this is a client getting data from the USB interface of the camera, for the OptiTrack Motion Capture, this is a client getting data from the network interface of the Motive software.

### For the IDS Camera
In order for the underlying IDS Peak API to work, you need to have the IDS Peak SDK installed on your system. The steps to install the SDK are as follows:
1. Go to the IDS Imaging website: [IDS Peak Download](https://en.ids-imaging.com/download-peak.html)
2. Choose your operating system (instructions below are for Linux).
3. Download the appropriate package:
   - For Linux 64-bit, select "IDS Peak xxx for Linux 64-bit – Debian package (without uEye Transport Layer)".
4. Install the package using the command:
   ```bash
   sudo apt install ./ids-peak-_xxx_amd64.deb
   ```  

### For the OptiTrack Motion Capture
#### Motive Software
You need to have the OptiTrack Motive software installed on a PC in the same network as the camera.
In Motive, apply the following settings:
1. Perform camera calibration and set the ground plane in Motive.
2. Enable streaming:
   - Go to View → Data Streaming Pane to open the streaming pane on the right.
3. Configure the settings:
   - Broadcast Frame Data: On
   - Local Interface: Set to the Motive-PC's IP (e.g., `172.22.147.182`)
   - Rigid Bodies: On
   - Transmission Type: Unicast
   - Multicast Interface: Same as Local Interface (e.g., `172.22.147.182`)
   - VRPN Broadcast Port: Leave as it is (default is `3883`)

#### NatNet SDK Python Client
In order for the NatNet SDK Python Client to work, you need to have the NatNet SDK installed on your system.
By cloning this repository, you already have a copy of the NatNet SDK Python Client in the `streams/NatNetSDK` directory, so nothing to do here.
However, if you download the SDK yourself, you need to make some modifications to the NatNet SDK Python Client to ensure it works correctly, so here are the steps for that:
1. Download and extract the NatNet SDK from [OptiTrack Developer Tools](https://optitrack.com/support/downloads/developer-tools.html#natnet-sdk).
2. The Python API consists of the modules in `./samples/PythonClient` (`DataDescriptions.py`, `MoCapData.py`, `NatNetClient.py`).
3. In order for the NatNet Python Client to close the streams properly, make the following changes to `NatNetClient.py`:
   - In the `run()` method, set `data_thread` and `command_thread` as daemon threads:
     ```python
     self.data_thread.daemon = True
     self.command_thread.daemon = True
     ```
   - In the `shutdown()` method, set timeout to 2 seconds:
     ```python
     self.command_thread.join(timeout=2)
     self.data_thread.join(timeout=2)
     ```
4. To return quality metrics of the rigid body poses, modify the `__unpack_rigid_body()` method:
   - Comment out the line:
     ```python
     if self.rigid_body_listener is not None: self.rigid_body_listener(new_id, pos, rot)
     ```
   - Replace it with:
     ```python
     if self.rigid_body_listener is not None: self.rigid_body_listener(new_id, pos, rot, marker_error, tracking_valid)
     ```

## Usage
### IDSStream Class
The `IDSStream` class is a wrapper around the IDS Peak API, allowing for easy access to camera functionalities. 
To use the `IDSStream` class, you can create an instance of it and call its methods to control the camera. For example:
```python
from streams.ids_stream import IDSStream 
cam_stream = IDSStream(frame_rate=30, 
                        exposure_time=20000, 
                        resize=(1000, 1000))
frame, info = cam_stream.getnext()
cam_stream.stop()
```
### MoCapStream Class
The `MoCapStream` class is a wrapper around the NatNet SDK Python Client, which allows for easy access to motion capture data from the OptiTrack Motive software.
To use the `MoCapStream` class, you can create an instance of it and call its methods to control the motion capture stream.
The motion capture data is hold in a buffer, to allow for interpolation of poses.
```python
from streams.mocap_stream import MoCapStream
mocap_stream = MoCapStream(client_ip="172.22.147.168",
                            server_ip="172.22.147.182", 
                            rigid_body_id=2,
                            buffer_size=20)
pose = mocap_stream.getnext()
mocap_stream.stop()
```

### Simultaneous Capture
This repository also provides functionality to capture data from both the IDS camera and the OptiTrack motion capture system simultaneously and match them based on timestamps.
A possible use case is to capture images from the IDS camera and corresponding poses from the OptiTrack system, which can then be used for various applications such as 3D reconstruction.
The intended pipeline for that is the following:
1. Start both streams:
```python
from streams.ids_stream import IDSStream
from streams.mocap_stream import MoCapStream
cam_stream = IDSStream(frame_rate=30, 
                        exposure_time=20000, 
                        resize=(1000, 1000))
mocap_stream = MoCapStream(client_ip="172.22.147.168",
                            server_ip="172.22.147.182", 
                            rigid_body_id=2,
                            buffer_size=20)
```
2. Start timing for both streams:
This resets the internal timers of both streams, allowing you to capture data with synchronized timestamps.
```python
cam_stream.start_timing()
mocap_stream.start_timing()
```
3. Take an image from the camera:
```python
frame, info = cam_stream.getnext()
timestamp = info['timestamp']
```
4. Get the corresponding pose from the motion capture system:
This method retrieves the pose at the specified timestamp, interpolating the pose buffer.
```python
pos, rot, v_trans, v_rot = mocap_stream.get_interpolated_pose(query_time=timestamp)
```

### Example Scripts
This repository contains several example scripts that demonstrate how to use the `IDSStream` and `MoCapStream` classes for various tasks:
- `capture_data.py`: Captures data from both the IDS camera and the OptiTrack motion capture system and visualizes the captured data.
- `capture_colmap_dataset.py`: Captures a dataset suitable for COLMAP, including images and corresponding poses.
- `check_time_sync.py`: Checks the time synchronization between the IDS camera and the OptiTrack motion capture system.
- `check_frame_rate.py`: Verifies the frame rates of both the IDS camera and the OptiTrack motion capture system.
Example usage:
```bash
python -m scripts.capture_data.py

