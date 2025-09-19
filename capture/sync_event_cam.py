import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
import scipy.interpolate as interp
from streams.ids_stream import IDSStream

def sync_event_cam(cam_stream):
    times = []
    rows = []

    while True:
        frame = cam_stream.getnext()
        if frame is None:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Threshold and erosion
        _, binary = cv2.threshold(gray, 254, 255, cv2.THRESH_BINARY)
        kernel = np.ones((25, 25), np.uint8)
        eroded = cv2.erode(binary, kernel, iterations=1)

        # Draw circles only on the RGB frame
        for c in cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
            if 5000 <= cv2.contourArea(c) <= 500000:
                (x, y), radius = cv2.minEnclosingCircle(c)
                cv2.circle(frame, (int(x), int(y)), int(radius), (0, 0, 255), 10) 
                times.append(time.time())
                rows.append(y)

        # Show
        eroded_bgr = cv2.cvtColor(eroded, cv2.COLOR_GRAY2BGR)
        combined = cv2.hconcat([eroded_bgr, frame])
        combined_resized = cv2.resize(combined, (1200, 600))
        cv2.imshow("Eroded | RGB", combined_resized)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam_stream.stop()
    cv2.destroyAllWindows()

    times = np.array(times)
    rows = np.array(rows)

    _, unique_indices = np.unique(rows, return_index=True)
    unique_indices.sort()
    unique_rows = rows[unique_indices]
    unique_times = times[unique_indices]

    interest_idx = np.argmin(unique_rows)
    start_idx = interest_idx - 10
    end_idx = interest_idx + 11

    selected_rows = unique_rows[start_idx:end_idx]
    selected_times = unique_times[start_idx:end_idx]

    spline = interp.UnivariateSpline(selected_times, selected_rows, k=3, s=0)

    time_fine = np.linspace(selected_times[0], selected_times[-1], 200)
    rows_fine = spline(time_fine)

    interest_idx = np.argmin(rows_fine)
    interest_row = rows_fine[interest_idx]
    interest_time = time_fine[interest_idx]

    intervals = np.diff(selected_times)
    avg_interval = np.mean(intervals)
    avg_rate = 1 / avg_interval
    interval_text = f"Avg interval: {avg_interval*1000:.2f} ms\nApprox. rate: {avg_rate:.2f} Hz"

    # Plotting
    plt.figure(figsize=(8,4))
    plt.plot(selected_times, selected_rows, 'o', label="Original points")
    plt.plot(time_fine, rows_fine, '-', label="Cubic spline")
    plt.plot(interest_time, interest_row, 'rx', markersize=10, label=f"Min row")
    plt.xlabel("Time (s)")
    plt.ylabel("Row coordinate (px)")
    plt.gca().invert_yaxis()
    plt.grid(True)
    plt.legend()
    plt.text(0.95, 0.05, interval_text, transform=plt.gca().transAxes,
            fontsize=10, color='blue', ha='right', va='bottom',
            bbox=dict(facecolor='white', alpha=0.6, edgecolor='gray'))
    plt.show()

    return interest_time

if __name__ == "__main__":
    cam_stream = IDSStream(frame_rate=45,
                           exposure_time='auto',
                           white_balance='auto',
                           gain='auto',
                           gamma=1.0)
    
    sync_event_time = sync_event_cam(cam_stream)

    print(f"Synchronized event camera time: {sync_event_time}")
