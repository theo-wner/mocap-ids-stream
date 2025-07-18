import cv2
import numpy as np
from packaging import version

def findChessboardCorners(image, chessboard, visualize=False):
    """
    Wrapper function for the OpenCV-Function findChessboardCornersSB() found at: https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#gadc5bcb05cb21cf1e50963df26986d7c9
    Simplifies the usage of above function by directly returning unique IDs for each corner based on the used chessboard.
    Only works with OpenCV Version ∈ [3.4, 4.11]
    Numpy 2 only works with OpenCV >= 4.10.0.84, so essentially the only two compatible OpenCV-Versions are [4.10.0.84, 4.11.0.86]
    The chessboard can be defined using the dictionary below.

    When working with the Radon Checkerboard, please note:
        - The three "dotted" squares mark the finder pattern
        - There must be one white square with black dot and two black squares with white dot that define a triangle
        - Of those three markers, the white square with the black dot marks the origin
        - Orient the chessboard so that the origin is top left, with the other two markers defining the axis down and the axis right
        - Each square belongs to its top left corner
        - When determining the position of a square: Count the number of corner in the desired direction beginning at 0

    Args:
        image (numpy.ndarray):
        chessboard (dict): A dictionary representing the used chessboard
            Expected Keys:
                - 'num_corners_down' (int): Count of corners in down-direction
                - 'num_corners_down' (int): Count of corner in right-direction
                - 'origin_marker_pos_down' (int): Position of the origin marker in down-direction
                - 'origin_marker_pos_right' (int): Position of the origin marker in right-direction
    """
    # Check if the current OpenCV Version works fine with findChessboardCornersSB()
    current_version = cv2.__version__
    if not (version.parse("4.3.0") <= version.parse(current_version) <= version.parse("4.11.0")):
        print(f"⚠️ Warning: OpenCV version {current_version} is not supported. Please install either 4.10.0.84 or 4.11.0.86")

    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Initial minimal pattern size --> Any patterns larger than this can be detected, so set very low
    initial_pattern_size = (3, 3)

    # Detection flags
    flags = (cv2.CALIB_CB_NORMALIZE_IMAGE | # Normalizes the input image histogram
            cv2.CALIB_CB_EXHAUSTIVE | # Improves Detection rate
            cv2.CALIB_CB_ACCURACY | # Improces Accuracy
            cv2.CALIB_CB_LARGER | # Allows finding more markers than given in the initial pattern size (needed for partially unvisible chessboard)
            cv2.CALIB_CB_MARKER) # Forces the input image to have the finder pattern (needed for identifying homologous points with partially unvisible chessboard)
    
    # Detect corners
    retval, corners, meta = cv2.findChessboardCornersSBWithMeta(grey, initial_pattern_size, flags)

    # Only continue if corners were found
    if retval:
        print(f"Total corners found: {len(corners)}")
        print(f"Detected pattern size: {meta.shape}")

        # Get the associated id for each corner
        ids = get_corner_ids(meta, chessboard)

        if visualize:
            # Annotate each corner with its ID
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            font_color = (0, 255, 0)
            thickness = 2

            for idx, corner in enumerate(corners):
                x, y = corner.ravel().astype(int)
                text = str(ids[idx])
                cv2.putText(image, text, (x + 5, y - 5), font, font_scale, font_color, thickness, cv2.LINE_AA)
            
            # Draw only the correct grid size
            cv2.drawChessboardCorners(image, meta.shape, corners, retval)

            # Show result
            scale = 0.4
            resized_img = cv2.resize(image, (0, 0), fx=scale, fy=scale)
            cv2.imshow('Chessboard Corners with IDs (Scaled)', resized_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return retval, corners, ids
    
    else:
        print("No chessboard corners found.")
        return False, None, None

def get_corner_ids(meta, chessboard):
    """
    Calculates an ID for each corner based on the used chessboard by
        1. Padding the detected meta array to align the local origin (value 4) to the global chessboard origin.
        2. Assigning an ID to the padded detected meta array
        3. Kicking out the non visible corners

    Args:
        meta (numpy.ndarray): Metadata returned by the function findChessboardCornersSB(). Contains info about wheather a corner is the top left corner of a square that is 
            black (1), white (2), black with white dot (3), white with black dot (4) or no meta data attached (0)
        chessboard (dict): See description in the function findChessboardCorners below
    """
    # Find the location of the origin in meta (local origin) --> either black or white dot
    loc = np.argwhere(meta == 4)
    if loc.shape[0] != 1:
        raise ValueError(f"Expected exactly one origin marker with ID 4 in meta. Please check that your chessboard has one black dot and two white dots")
    
    origin_down_local, origin_right_local = loc[0]

    # Where the origin (4) should be in the full chessboard
    origin_down_global = chessboard['origin_marker_pos_down']
    origin_right_global = chessboard['origin_marker_pos_right']

    # Compute how much padding is needed before and after
    pad_top = origin_down_global - origin_down_local
    pad_left = origin_right_global - origin_right_local
    pad_bottom = chessboard['num_corners_down'] - (pad_top + meta.shape[0])
    pad_right = chessboard['num_corners_right'] - (pad_left + meta.shape[1])
    print(pad_bottom, pad_left, pad_right, pad_top)

    # Sanity check
    if pad_top < 0 or pad_bottom < 0 or pad_left < 0 or pad_right < 0:
        raise ValueError("Meta data too large or origin mismatch for target chessboard size. Please check your chessboard definition")

    # Pad with -1 to fill the full chessboard
    padded_meta = np.pad(meta,
                         ((pad_top, pad_bottom), (pad_left, pad_right)),
                         mode='constant',
                         constant_values=9)
    
    print(padded_meta)
    
    # Create ID array of the whole chessboard
    paddded_ids = np.arange(chessboard['num_corners_down'] * chessboard['num_corners_right']).reshape((chessboard['num_corners_down'] , chessboard['num_corners_right']))

    # Kick out non visible entries
    ids = paddded_ids[padded_meta != 9]
    return ids

if __name__ == '__main__':
    # Load image
    image = cv2.imread('./data/tmp_colmap_dataset/images/0001.png')

    # Define Chessboard --> Information in description of findChessboardCorners
    chessboard = {'num_corners_down' : 14,
                  'num_corners_right' : 9,
                  'origin_marker_pos_down' : 5,
                  'origin_marker_pos_right' : 3}
    
    # Find corners
    retval, corners, ids = findChessboardCorners(image, chessboard, visualize=True)





