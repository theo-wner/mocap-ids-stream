import cv2
import numpy as np

def get_corner_ids(meta):
    """
    Takes the metadata returned by cv2.findChessboardCornersSBWithMeta and calculates the id of 
    each corner based on the coordinate system in meta with its origin defined by the finder pattern
    """
    # Flatten and find index of the origin (value 4)
    flat = meta.flatten()
    origin_index = np.where(flat == 4)[0][0]

    # Create ID map relative to origin
    ids = np.arange(len(flat)) - origin_index + 1000
    return ids

if __name__ == '__main__':
    # Load image
    img = cv2.imread('./colmap_dataset/images/0000.png')
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Initial minimal pattern size
    initial_pattern_size = (3, 3)

    # Detection flags
    flags = (cv2.CALIB_CB_NORMALIZE_IMAGE | # Normalizes the input image histogram
            cv2.CALIB_CB_EXHAUSTIVE | # Improves Detection rate
            cv2.CALIB_CB_LARGER | # Allows finding more markers than given in the initial pattern size (needed for partially unvisible chessboard)
            cv2.CALIB_CB_MARKER) # Forces the input image to have the three marker dots --> finder pattern (needed for identifying homologous points with partially unvisible chessboard)

    # Detect corners
    retval, corners, meta = cv2.findChessboardCornersSBWithMeta(grey, initial_pattern_size, flags)

    if retval:
        detected_pattern_size = meta.shape
        print("Chessboard detected.")
        print(f"Total corners found: {len(corners)}")
        print(f"Detected pattern size: {detected_pattern_size}")

        ids = get_corner_ids(meta)

        # Annotate each corner with its ID
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_color = (0, 255, 0)
        thickness = 2

        for idx, corner in enumerate(corners):
            x, y = corner.ravel().astype(int)
            text = str(ids[idx])
            cv2.putText(img, text, (x + 5, y - 5), font, font_scale, font_color, thickness, cv2.LINE_AA)

        # Draw only the correct grid size
        cv2.drawChessboardCorners(img, detected_pattern_size, corners, retval)

        # Show result
        scale = 0.4
        resized_img = cv2.resize(img, (0, 0), fx=scale, fy=scale)
        cv2.imshow('Chessboard Corners with IDs (Scaled)', resized_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    else:
        print("Chessboard corners not found.")