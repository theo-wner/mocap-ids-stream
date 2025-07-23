import numpy as np
import os
import cv2
from hec.calib_utils import findChessboardCorners

# Define Chessboard
chessboard = {'num_corners_down' : 23,
                'num_corners_right' : 16,
                'origin_marker_pos_down' : 10,
                'origin_marker_pos_right' : 7,
                'square_size' : 16}
cd = chessboard['num_corners_down']
cr = chessboard['num_corners_right']
ss = chessboard['square_size']

# Define Object Points in millimeters
objp = np.zeros((cd*cr, 3), np.float32)
objp[:,:2] = np.mgrid[0:cr*ss:ss, 0:cd*ss:ss].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d point in image plane

image_folder = "./data/chessboard"

for filename in os.listdir(image_folder):
    filepath = os.path.join(image_folder, filename)
    img = cv2.imread(filepath)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find chess board corners
    retval, found_shape, corners, ids = findChessboardCorners(gray, chessboard, visualize=False)

    # If found, add the object points and image points
    if retval:
        visible_objp = np.array([objp[i] for i in ids])
        objpoints.append(visible_objp)
        imgpoints.append(corners)

        # Draw the corners with their ids
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_color = (0, 255, 0)
        thickness = 2
        for idx, corner in enumerate(corners):
            x, y = corner.ravel().astype(int)
            text = str(ids[idx])
            cv2.putText(img, text, (x + 5, y - 5), font, font_scale, font_color, thickness, cv2.LINE_AA)
        cv2.drawChessboardCorners(img, found_shape, corners, retval)
        scale = 0.3
        resized_img = cv2.resize(img, (0, 0), fx=scale, fy=scale)
        cv2.imshow('img', resized_img)
        cv2.waitKey(100)

cv2.destroyAllWindows()

# Calibrate
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# Calculate mean reprojection error
mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    mean_error += error
print( "total error: {}".format(mean_error/len(objpoints)) )

# Save calibration result
np.savez('calibration_result.npz', ret=ret, mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)