import numpy as np
import cv2
from datetime import datetime
import os

"""
CONFIG
"""
CAMERA_DEVISE_L = 0
CAMERA_DEVISE_R = 1

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480

SAVED_FOLDER = 'images'
"""
"""
if not os.path.exists(SAVED_FOLDER):
    os.mkdir(SAVED_FOLDER)


capl = cv2.VideoCapture(CAMERA_DEVISE_L)
capl.set(cv2.CAP_PROP_FRAME_WIDTH, IMAGE_WIDTH)
capl.set(cv2.CAP_PROP_FRAME_HEIGHT, IMAGE_HEIGHT)

capr = cv2.VideoCapture(CAMERA_DEVISE_R)
capr.set(cv2.CAP_PROP_FRAME_WIDTH, IMAGE_WIDTH)
capr.set(cv2.CAP_PROP_FRAME_HEIGHT, IMAGE_HEIGHT)

while True:

    if not (capl.grab() and capr.grab()):
        print("No more frames")
        break

    ret, framel = capl.read()
    ret, framer = capr.read()

    # 画面に表示する
    cv2.imshow('left', framel)
    cv2.imshow('right', framer)

    key = cv2.waitKey(1) & 0xFF

    # q or esc が押された場合は終了する
    if key == ord('q') or key == 27:
        break

    # sが押された場合は保存する
    if key == ord('s'):
        filename_base = datetime.now().strftime("%Y%m%d-%H%M%S")
        pathl = "images/" + filename_base + "-Left.jpg"
        pathr = "images/" + filename_base + "-Right.jpg"
        cv2.imwrite(pathl, framel)
        cv2.imwrite(pathr, framer)
        print("[NOTE] Save file. {}".format(pathl))
        print("[NOTE] Save file. {}".format(pathr))

# キャプチャの後始末と，ウィンドウをすべて消す
capl.release()
capr.release()
cv2.destroyAllWindows()
