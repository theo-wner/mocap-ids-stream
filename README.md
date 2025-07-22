# MoCap-IDS-Stream - Documentation

This repository provides a framework for capturing and processing camera and motion capture data using the IDS U3-31J0CP camera and the OptiTrack Motive software. 
It includes the classes `IDSStream` and `MoCapStream` for handling IDS camera and OptiTrack motion capture data streams.
This is aimed towards the application of connecting a camera to a rigid body, trackable with a motion capture system.
For determining the relative Pose between the Camera Coordinate System and the Rigid Body Coordinate System, this repository features functionality for performing a Hand-Eye-Calibration (HEC).
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
4. Create new Rigid Body for the camera rig under Layout → Create

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

## Setting up this repository
### Directly working with this repository
To get started with working with this repository directly, clone this repository and create a new conda environment using
```bash
conda create -n mocap_ids python=3.12
conda activate mocap_ids
pip install -r requirements.txt
```
Additionally, install the [PyTorch version that best fits your system](https://pytorch.org/get-started/locally/).

### Using this repository as a submodule
To use this repository as a git submodule inside another project, add the submodule and edit .gitmodules (Here, the submodule gets added into a directory called "submodules"):
```bash
git submodule add git@github.com:theo-wner/mocap-ids-stream.git submodules
git add .gitmodules submodules
git commit -m "Add submodule"
```
Then install the submodule via pip:
```bash
pip install submodules/mocap-ids-stream
```
Pull changes via:
```bash
git submodule update --remote submodules/mocap-ids-stream
```
and reinstall with pip

## Usage
### IDSStream Class
The `IDSStream` class is a wrapper around the IDS Peak API, allowing for easy access to camera functionalities. 
To use the `IDSStream` class, you can create an instance of it and call its methods to control the camera. For example:
```python
from streams.ids_stream import IDSStream 
   cam_stream = IDSStream(frame_rate='max', 
                        exposure_time='auto', 
                        white_balance='auto',
                        gain='auto',
                        gamma=1.0,
                        resize=None)
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
                              buffer_size=15)
pose = mocap_stream.getnext()
mocap_stream.stop()
```
### StreamMatcher Class
This repository also provides functionality to capture data from both the IDS camera and the OptiTrack motion capture system simultaneously and match them based on timestamps.
This is handled by the `StreamMatcher` class.
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
2. Initialize and start StreamMatcher:
This starts the StreamMatcher instance, providing functionality for simultaneous Camera and MoCap data retrieval.
Also, the StreamMatcher insance handles time synchronization by resetting the internal timers of both streams with continouus resnychronizing.
```python
matcher = StreamMatcher(cam_stream, mocap_stream, resync_interval=10)
matcher.start_timing()
```
3. Retrieve simultaneous Camera and MoCap data:
This retrieves the most current camera frame and calculates the best fitting interpolated pose being returned in `info`.
```python
frame, info = matcher.getnext()
```
4. Stop all running Stream instances:
```python
matcher.stop()
cam_stream.stop()
mocap_stream.stop()
```

### Example Scripts
This repository contains several capture scripts that demonstrate how to use the `IDSStream`, `MoCapStream` and `StreamMatcher` classes for various tasks:
- `visualize_data.py`: Captures data from both the IDS camera and the OptiTrack motion capture system and visualizes the captured data.
- `capture_dataset.py`: Captures a dataset, including images and corresponding poses.
It also features debugging scripts useful for testing:
- `check_time_sync.py`: Checks the time synchronization between the IDS camera and the OptiTrack motion capture system.
- `check_frame_rate.py`: Verifies the frame rates of both the IDS camera and the OptiTrack motion capture system.
Example usage:
```bash
python -m scripts.capture_data.py
```

