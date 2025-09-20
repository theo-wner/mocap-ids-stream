# MoCap-IDS-Stream - Documentation

This repository provides a framework for capturing and processing camera and motion capture data using the IDS U3-31J0CP camera and the OptiTrack Motive software. 
It includes the classes `IDSStream` and `MoCapStream` for handling IDS camera and OptiTrack motion capture data streams.
Additionally it includes the class `StreamMatcher` for handling simultaneous capture of both streams.
This is aimed towards the application of connecting a camera to a rigid body, trackable with a motion capture system.
For determining the intrinsics of the camera as well as relative Pose between the Camera Coordinate System and the Rigid Body Coordinate System, this repository features functionality for performing a Camera Calibration and a Hand-Eye-Calibration (HEC).

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
4. Pull the print-statement in line ~2080 into the if condition

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
git submodule add git@github.com:theo-wner/mocap-ids-stream.git submodules/mocap-ids-stream
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
   cam_stream = IDSStream(frame_rate=45, 
                        exposure_time='auto', 
                        white_balance='auto',
                        gain='auto',
                        gamma=1.0)
frame, info = cam_stream.getnext()
cam_stream.stop()
```
### MoCapStream Class
The `MoCapStream` class is a wrapper around the NatNet SDK Python Client, which allows for easy access to motion capture data from the OptiTrack Motive software.
To use the `MoCapStream` class, you can create an instance of it and call its methods to control the motion capture stream.
The motion capture data is hold in a buffer.
```python
from streams.mocap_stream import MoCapStream
   mocap_stream = MoCapStream(client_ip="172.22.147.168",
                              server_ip="172.22.147.182",
                              buffer_size=20)
pose = mocap_stream.getnext()
mocap_stream.stop()
```
### StreamMatcher Class
This repository also provides functionality to capture data from both the IDS camera and the OptiTrack motion capture system simultaneously and match them.
This is handled by the `StreamMatcher` class.
A possible use case is to capture images from the IDS camera and corresponding poses from the OptiTrack system, which can then be used for various applications such as 3D reconstruction.
The intended pipeline for that is the following:
1. Start both streams:
```python
from streams.ids_stream import IDSStream
from streams.mocap_stream import MoCapStream
cam_stream = IDSStream(frame_rate=30, 
                        exposure_time=20000)
mocap_stream = MoCapStream(client_ip="172.22.147.168",
                            server_ip="172.22.147.182",
                            buffer_size=20)
```
2. Initialize and start StreamMatcher:
This starts the StreamMatcher instance, providing functionality for simultaneous Camera and MoCap data retrieval.
```python
matcher = StreamMatcher(cam_stream, mocap_stream, rb_id=2, calib_path=None, downsampling=None)
```
3. Retrieve simultaneous Camera and MoCap data:
This retrieves the most current camera frame and the best fitting MoCap pose stored in `info`.
```python
frame, info = matcher.getnext()
```
4. Stop all running Stream instances:
```python
cam_stream.stop()
mocap_stream.stop()
```

### Time synchronisation
To account for the different latencies of camera and MoCap data, a fixed offset is assumed and applied inside the `StreamMatcher` class.
To obtain this latency for the specific system setup, the script `monitor_latency.py` is provided.
This involves an experiment inside the working space, which can be observed by both camera and MoCap. For further information, look into the documentation inside the script.

### Calibration
In order to directly obtain the Pose of the MoCap Base CS w.r.t. the CCS, the transform between the Rigid Body Coordinate System (hereby referred es Tool Coordinate System, TCS) and the CCS has to be calculated via a Hand-Eye-Calibration (HEC).
To do that, the intended pipeline in this repository is the follwing:
1. Execute `capture_calibration_dataset.py`:
This creates a new directory under `/data`. You can specify its name via the `--calib_path`-flag or set `--calib_path` to `default` to obtain a timestamp-based directory name.
With this script, you can then capture images of an OpenCV-readable chessboard, which can later be used for camera calibration. Additionally, the MoCap poses of all images are saved under `images_mocap.txt` for the HEC later.

2. Filter out blurry images:
All images are saved inside the `/images` subdirectory. You can now delete the ones not suitable for the calibration. There is no need to delete the corresponding MoCap poses.

3. Execute `perform_calibration.py`:
This script first performs an OpenCV-based camera calibration of the calibration directory specified with `--calib_path`, or of the most recent one when setting `--calib_path` to `latest`.
All calvulated image poses are then saved under `images_checkerboard.txt` and the calculated intrinsics under `intrinsics.txt`
Sequentially, the OpenCV-based HEC is performed using pairs of MoCap poses and checkerboard-based poses. The linearly calculated hand-eye-pose is then saved to `hand_eye_pose.txt`.
The calculated Hand-Eye-Pose and Base-World-Pose are then refined using a nonlinear approach by minimizing the reprojection error.

4. Now the calibration process is completed and the path to the calibration directory can be passed to a StreamMatcher-Object, which then automatically applies the hand-eye-pose to the MoCap poses it returnes in the `getnext`-function. 
You can either specify the path to the calibration dataset directly or pass `latest` to use the latest calibration.
```python
matcher = StreamMatcher(cam_stream, mocap_stream, rb_id=2, calib_path="latest", downsampling=None)
```
Alternatively, a COLMAP-Style dataset can now be captured using the script `capture_calibrated_dataset.py`, which again takes the `--calib_path`-flag pointing to the desired calibration directory or the `latest` calibration.
Additionally, this script now takes the flag `--dataset_path`, pointing to the desired path of the dataset to capture. This flag can again be a custom path or `deafult` to obtain a timestamp-based directory name.
This script automatically applies the optimized Hand-Eye-Pose of the given `--calib_path`.
