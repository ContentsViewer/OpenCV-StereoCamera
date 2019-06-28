import numpy as np
import cv2
import glob
import random
import json
import sys
import os

from lib.extended_json import ExtendedJsonEncoder


"""
CONFIG
"""
PATTERN_SIZE = (7, 10)
IMAGE_PATH = "images/*"
LEFT_IMAGE_SUFFIX = "-Left.jpg"
RIGHT_IMAGE_SUFFIX = "-Right.jpg"
CALIBRATION_DATA_FILENAME = 'stereo_calibration_data.json'

"""
"""

calibration_data = {}

imagesl = glob.glob(IMAGE_PATH + LEFT_IMAGE_SUFFIX)
imagesr = glob.glob(IMAGE_PATH + RIGHT_IMAGE_SUFFIX)

image_set = []

# print(imagesl)
# print(imagesr)

for imagel in imagesl:
    print("matching... left image: " + imagel)
    basename = imagel[0: -len(LEFT_IMAGE_SUFFIX)]
    if basename + RIGHT_IMAGE_SUFFIX in imagesr:
        image_set.append((basename + LEFT_IMAGE_SUFFIX,
                          basename + RIGHT_IMAGE_SUFFIX))
        print(
            "matched. ({}) <-> ({})".format(image_set[-1][0], image_set[-1][1]))
    else:
        print("matching fail.")

#
#
#

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((PATTERN_SIZE[0] * PATTERN_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:PATTERN_SIZE[0], 0:PATTERN_SIZE[1]].T.reshape(-1, 2)


#
#
#

objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.


print("[Info] Starting Left camera calibration...")
for imagel in imagesl:
    print("read image file {}".format(imagel))

    img = cv2.imread(imagel)
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

ret, mtxl, distl, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None)

img = cv2.imread(random.choice(imagesl))
h, w = img.shape[:2]
newcameramtxl, roi = cv2.getOptimalNewCameraMatrix(
    mtxl, distl, (w, h), 1, (w, h))

calibration_data['mtxl'] = mtxl
calibration_data['distl'] = distl
calibration_data['newcameramtxl'] = newcameramtxl

print("mtxl: {}".format(mtxl))
print("distl: {}".format(distl))
print("newcameramtxl: {}".format(newcameramtxl))

# undistort
dst = cv2.undistort(img, mtxl, distl, None, newcameramtxl)
cv2.imshow('calibresult', dst)


cv2.waitKey(0)


#
#
#

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.


print("[Info] Starting Right camera calibration...")
for imager in imagesr:
    print("read image file {}".format(imager))

    img = cv2.imread(imager)
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

ret, mtxr, distr, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None)

img = cv2.imread(random.choice(imagesr))
h, w = img.shape[:2]
newcameramtxr, roi = cv2.getOptimalNewCameraMatrix(
    mtxr, distr, (w, h), 1, (w, h))


calibration_data['mtxr'] = mtxr
calibration_data['distr'] = distr
calibration_data['newcameramtxr'] = newcameramtxr

print("mtxr: {}".format(mtxr))
print("distr: {}".format(distr))
print("newcameramtxr: {}".format(newcameramtxr))

# undistort
dst = cv2.undistort(img, mtxr, distr, None, newcameramtxr)
cv2.imshow('calibresult', dst)

cv2.waitKey(0)


#
#
#

objpoints = []  # 3d point in real world space
imgpointsl = []  # 2d points in image plane.
imgpointsr = []  # 2d points in image plane.

for image_lr in image_set:

    print("read image L file {}".format(image_lr[0]))
    print("read image R file {}".format(image_lr[1]))

    im_l = cv2.imread(image_lr[0])
    gray_l = cv2.cvtColor(im_l, cv2.COLOR_BGR2GRAY)
    im_r = cv2.imread(image_lr[1])
    gray_r = cv2.cvtColor(im_r, cv2.COLOR_BGR2GRAY)

    # コーナー検出
    found_l, corners_l = cv2.findChessboardCorners(gray_l, PATTERN_SIZE, None)
    found_r, corners_r = cv2.findChessboardCorners(gray_r, PATTERN_SIZE, None)

    # コーナーがあれば
    if found_l and found_r:

        objpoints.append(objp)

        cv2.cornerSubPix(gray_l, corners_l, (11, 11), (-1, -1), criteria)
        cv2.cornerSubPix(gray_r, corners_r, (11, 11), (-1, -1), criteria)

        imgpointsl.append(corners_l)
        imgpointsr.append(corners_r)

        cv2.drawChessboardCorners(im_l, PATTERN_SIZE, corners_l, found_l)
        cv2.drawChessboardCorners(im_r, PATTERN_SIZE, corners_r, found_r)

        cv2.imshow('img L', im_l)
        cv2.imshow('img R', im_r)
        cv2.waitKey(500)

retval, cameramatrixl, distcoeffsl, cameramatrixr, distcoeffsr, R, T, E, F = cv2.stereoCalibrate(
    objpoints, imgpointsl, imgpointsr, newcameramtxl, distl, newcameramtxr, distr, gray_l.shape[::-1])


print("cameramatrixl: {}".format(cameramatrixl))  # 新しいカメラ行列(L)
print("cameramatrixr: {}".format(cameramatrixr))  # 新しいカメラ行列(R)
print("distcoeffsl: {}".format(distcoeffsl))   # 新しいゆがみ係数(L)
print("distcoeffsr: {}".format(distcoeffsr))  # 新しいゆがみ係数(R)
print("R: {}".format(R))  # カメラ間回転行列
print("T: {}".format(T))  # カメラ間並進ベクトル

cv2.waitKey(0)

calibration_data['cameramatrixl'] = cameramatrixl
calibration_data['cameramatrixr'] = cameramatrixr
calibration_data['distcoeffsl'] = distcoeffsl
calibration_data['distcoeffsr'] = distcoeffsr
calibration_data['R'] = R
calibration_data['T'] = T

fw = open(CALIBRATION_DATA_FILENAME, 'w')
json.dump(calibration_data, fw, indent=4, cls=ExtendedJsonEncoder)

print("Write Calibration Data in File {}".format(CALIBRATION_DATA_FILENAME))
print("Completely task edned.")
