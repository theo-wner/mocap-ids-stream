import threading
import matplotlib.pyplot as plt

class StreamMatcherDebug:
    def __init__(self, ids_stream, mocap_stream):
        self.ids_stream = ids_stream
        self.mocap_stream = mocap_stream

    def get_time_diff(self):
        stop_event = threading.Event()
        mocap_results = {}

        # Run MoCap sync in a separate thread (no GUI in this thread)
        def mocap_runner():
            mocap_results.update(self.mocap_stream.sync_event(stop_event=stop_event))

        mocap_thread = threading.Thread(target=mocap_runner)
        mocap_thread.start()

        # Run camera sync in main thread
        ids_results = self.ids_stream.sync_event()

        # Stop MoCap acquisition
        stop_event.set()
        mocap_thread.join()

        # Extract times
        ids_time = ids_results["interest_time"]
        mocap_time = mocap_results["interest_time"]
        time_diff = (ids_time - mocap_time) * 1000  # ms

        # -------------------------
        # Plot results
        # -------------------------
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # IDS
        ax1.plot(ids_results["original_times"], ids_results["original_rows"], 'o', label="Original Points")
        ax1.plot(ids_results["interp_times"], ids_results["interp_rows"], '-', label="Cubic spline")
        ax1.plot(ids_time, ids_results["interest_row"], 'rx', markersize=10, label="Min Row")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Row coordinate (px)")
        ax1.invert_yaxis()
        ax1.grid(True)
        ax1.legend()
        ax1.set_title("Cam Sync Event")

        # Add sampling info inline
        ids_interval = ids_results["original_times"][-1] - ids_results["original_times"][0]
        ids_samples = len(ids_results["original_times"])
        ids_fps = ids_samples / ids_interval if ids_interval > 0 else 0
        ids_dt_ms = (ids_interval / (ids_samples - 1) * 1000) if ids_samples > 1 else 0
        ax1.text(0.95, 0.95, f"Sampling frequency: {ids_fps:.2f} Hz\nSampling interval: {ids_dt_ms:.2f} ms",
                 transform=ax1.transAxes, ha='right', va='top', fontsize=10, bbox=dict(facecolor='white', alpha=0.7))

        # MoCap
        ax2.plot(mocap_results["original_times"], mocap_results["original_ys"], 'o', label="Original Points")
        ax2.plot(mocap_results["interp_times"], mocap_results["interp_ys"], '-', label="Cubic spline")
        ax2.plot(mocap_time, mocap_results["interest_y"], 'rx', markersize=10, label="Max Y")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Y position")
        ax2.grid(True)
        ax2.legend()
        ax2.set_title("MoCap Sync Event")

        # Add sampling info inline
        mocap_interval = mocap_results["original_times"][-1] - mocap_results["original_times"][0]
        mocap_samples = len(mocap_results["original_times"])
        mocap_fps = mocap_samples / mocap_interval if mocap_interval > 0 else 0
        mocap_dt_ms = (mocap_interval / (mocap_samples - 1) * 1000) if mocap_samples > 1 else 0
        ax2.text(0.95, 0.95, f"Sampling frequency: {mocap_fps:.2f} Hz\nSampling interval: {mocap_dt_ms:.2f} ms",
                 transform=ax2.transAxes, ha='right', va='top', fontsize=10, bbox=dict(facecolor='white', alpha=0.7))

        plt.suptitle(f"Time difference (IDS - MoCap): {time_diff:.2f} ms\nPress 'k' to keep, 'r' to reject")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        # -------------------------
        # Wait for user decision
        # -------------------------
        decision = {"keep": None}

        def on_key(event):
            if event.key == "k":
                decision["keep"] = True
                plt.close(fig)
            elif event.key == "r":
                decision["keep"] = False
                plt.close(fig)

        fig.canvas.mpl_connect("key_press_event", on_key)
        plt.show()

        return time_diff if decision["keep"] else None
