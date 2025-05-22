# 3DGS with Mocap - Documentation

## Manual for working with the mocap-gs repo

### Connect IDS U3-31J0CP Rev.2.2 to workstation
- plug in usb-cable 

### Set up Motive on any PC in the same network
- perform camera calibration and set ground plane
- enable streaming: View --> Data Streaming Pane --> Streaming Pane appears on the right
- settings: 
    - Broadcast Frame Data: On
    - Local Interface: The Motive-PC's IP (currently 172.22.147.182)
    - Rigid Bodies: On
    - Transmission Type: Unicast
    - Multicast Interface: Same as Local Interface (currently 172.22.147.182)
    - VRPN Broadcast Port: leave as it is (3883)
 
### Working with the mocap-gs repo
- the module "data_streams" contains the wrapper-classes CameraStream (wrapping the IDS Peak API) and MoCapStream (wrapping the NatNetSDK Python Client)
- execute scripts via ```python -m scripts.my_script``` or
- play around with the CameraStream and MoCapStream classes

## Documentation for me: How to build the mocap-gs repo
### Set up Python client on the workstation
- download and extract NatNet SDK from https://optitrack.com/support/downloads/developer-tools.html#natnet-sdk
- the Python API consists of the modules in ./samples/PythonClient (DataDescriptions.py, MoCapData.py, NatNetClient.py)
- in order for the NatNet Python Client to close the streams properly the following changes have to be made to NatNetClient.py:
    - in the run()-Method: Set data_thread and command_thread as daemon threads via self.data_thread.daemon = True, self.command_thread.daemon = True
    - in the shutdown()-Method: set timeout to 2 seconds: self.command_thread.join(timeout=2), self.data_thread.join(timeout=2)
