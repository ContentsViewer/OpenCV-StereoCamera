import numpy as np
import cv2
import glob
import random
import json
import sys

from lib.extended_json import ExtendedJsonEncoder

"""
CONFIG
"""

PATTERN_SIZE = (7, 10)
IMAGE_PATH = "imagesl/*.jpg"
CALIBRATION_DATA_FILENAME = 'calibration_data.json'

"""
"""

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((PATTERN_SIZE[0] * PATTERN_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:PATTERN_SIZE[0], 0:PATTERN_SIZE[1]].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

images = glob.glob(IMAGE_PATH)

if len(images) <= 0:
    print(F"[NOTE] no files in {IMAGE_PATH}.")
    exit()

for fname in images:
    print("read image file {}".format(fname))

    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, PATTERN_SIZE, None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners)

        # Draw and display the corners
        cv2.drawChessboardCorners(img, PATTERN_SIZE, corners, ret)
        cv2.imshow('img', img)
        cv2.waitKey(500)

cv2.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None)

img = cv2.imread(random.choice(images))
h, w = img.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

calibration_data = {}
calibration_data['mtx'] = mtx
calibration_data['dist'] = dist
calibration_data['rvecs'] = rvecs
calibration_data['tvecs'] = tvecs
calibration_data['roi'] = roi
calibration_data['newcameramtx'] = newcameramtx

print("Camera Calibration Data:")
print(calibration_data)

# undistort
dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

# crop the image
x, y, w, h = roi
dst = dst[y: y + h, x: x + w]
cv2.imshow('calibresult', dst)

cv2.waitKey(0)

cv2.imwrite('calibresult.png', dst)


fw = open(CALIBRATION_DATA_FILENAME, 'w')
json.dump(calibration_data, fw, indent=4, cls=ExtendedJsonEncoder)

print(F"Write Calibration Data in File {CALIBRATION_DATA_FILENAME}")
print("Completely task ended.")
