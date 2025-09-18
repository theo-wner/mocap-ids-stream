from streams.ids_stream_debug import IDSStreamDebug
from streams.mocap_stream import MoCapStream
from streams.stream_matcher_debug import StreamMatcherDebug

cam_stream = IDSStreamDebug(frame_rate='max', 
                        exposure_time='auto', 
                        white_balance='auto',
                        gain='auto',
                        gamma=1.0)

mocap_stream = MoCapStream(client_ip="172.22.147.168", # 168 for workstation, 172 for laptop
                            server_ip="172.22.147.182", 
                            rigid_body_id=2, # 1 for calibration wand, 2 for camera rig
                            buffer_size=15)

matcher = StreamMatcherDebug(cam_stream, mocap_stream)
time_diff = matcher.get_time_diff()

print(time_diff)

cam_stream.stop()
mocap_stream.stop()
