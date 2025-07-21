"""
Module for matching data and handling time sync of an IDS Camera Stream and a OptiTrack MoCap Stream

Author:
    Theodor Kapler <theodor.kapler@student.kit.edu>
"""
import threading
import time

class StreamMatcher():
    """
    A class to handle Streams from both an IDS Camera and a OptiTrack MoCap System
    """
    def __init__(self, ids_stream, mocap_stream, resync_interval):
        self.ids_stream = ids_stream
        self.mocap_stream = mocap_stream
        self.resync_interval = resync_interval
        self.resync_thread = threading.Thread(target=self.resync_loop, daemon=True)
        self.running = True

    def start(self):
        self.ids_stream.start_timing()
        self.mocap_stream.start_timing()
        self.resync_thread.start()

    def resync_loop(self):
        while self.running:
            time.sleep(self.resync_interval)
            print("Resyncronizing timestamps")
            self.ids_stream.start_timing
            self.mocap_stream.start_timing

    def getnext(self):
        

    def stop(self):
        self.running = False
        self.resync_thread.join()