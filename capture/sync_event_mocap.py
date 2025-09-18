import numpy as np
import matplotlib.pyplot as plt
import time
import scipy.interpolate as interp
from streams.mocap_stream import MoCapStream

def sync_event_mocap(mocap_stream):
    times = []
    ys = []

    while True:
        try:
            mocap_data = mocap_stream.getnext()

            marker_list = list(mocap_data["labeled_markers"].values())

            single_markers = []
            for marker in marker_list:
                if marker["belongs_to_rb"] == False:
                    single_markers.append(marker)

            if len(single_markers) != 1:
                print(f"{len(single_markers)} single markers visible, not exactly 1")
                continue

            y = single_markers[0]["pos"][1]
            ys.append(y)
            times.append(time.time())

            print(f"y = {y:.2f}")  # live feedback
            time.sleep(0.0001)
        except KeyboardInterrupt:
            print(f"Collected {len(times)} mocap samples in {times[-1] - times[0]:.2f} s "
      f"(~{len(times) / (times[-1] - times[0]):.1f} Hz)")

            break

    mocap_stream.stop()

    times = np.array(times)
    ys = np.array(ys)

    _, unique_indices = np.unique(ys, return_index=True)
    unique_indices.sort()
    unique_ys = ys[unique_indices]
    unique_times = times[unique_indices]

    interest_idx = np.argmax(unique_ys)
    start_idx = interest_idx - 20
    end_idx = interest_idx + 21

    selected_ys = unique_ys[start_idx:end_idx]
    selected_times = unique_times[start_idx:end_idx]

    spline = interp.UnivariateSpline(selected_times, selected_ys, k=3, s=0)

    time_fine = np.linspace(selected_times[0], selected_times[-1], 200)
    ys_fine = spline(time_fine)

    interest_idx = np.argmax(ys_fine)
    interest_y = ys_fine[interest_idx]
    interest_time = time_fine[interest_idx]

    intervals = np.diff(selected_times)
    avg_interval = np.mean(intervals)
    avg_rate = 1 / avg_interval
    interval_text = f"Avg interval: {avg_interval*1000:.2f} ms\nApprox. rate: {avg_rate:.2f} Hz"

    # Plot
    plt.figure(figsize=(8, 4))
    plt.plot(selected_times, selected_ys, 'o', label="Original Points")
    plt.plot(time_fine, ys_fine, '-', label="Cubic spline")
    plt.plot(interest_time, interest_y, 'rx', markersize=10, label="Max y")
    plt.xlabel("Time (s)")
    plt.ylabel("Y coordinate")
    plt.grid(True)
    plt.legend()
    plt.text(0.95, 0.05, interval_text, transform=plt.gca().transAxes,
             fontsize=10, color='blue', ha='right', va='bottom',
             bbox=dict(facecolor='white', alpha=0.6, edgecolor='gray'))
    plt.show()

    return interest_time


if __name__ == "__main__":
    mocap_stream = MoCapStream(
        client_ip="172.22.147.168",  # 168 for workstation, 172 for laptop
        server_ip="172.22.147.182",
        rigid_body_id=2,  # 1 for calibration wand, 2 for camera rig
        buffer_size=15
    )

    sync_event_time = sync_event_mocap(mocap_stream)
    print(f"Synchronized mocap time: {sync_event_time}")
