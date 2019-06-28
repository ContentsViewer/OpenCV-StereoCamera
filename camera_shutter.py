import numpy as np
import cv2
from datetime import datetime
import os

"""
CONFIG
"""
CAMERA_DEVISE = 1

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
SAVED_FOLDER = 'images'

"""
"""
if not os.path.exists(SAVED_FOLDER):
    os.mkdir(SAVED_FOLDER)

cap = cv2.VideoCapture(CAMERA_DEVISE)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMAGE_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMAGE_HEIGHT)


while True:

    if not cap.grab():
        print("no more frame")
        break

    ret, frame = cap.read()

    # 画面に表示する
    cv2.imshow('frame', frame)

    key = cv2.waitKey(1) & 0xFF

    # q or esc が押された場合は終了する
    if key == ord('q') or key == 27:
        break

    # sが押された場合は保存する
    if key == ord('s'):
        path = SAVED_FOLDER + "/" + datetime.now().strftime("%Y%m%d-%H%M%S") + ".jpg"
        cv2.imwrite(path, frame)
        print("[NOTE] Save file. {}".format(path))

# キャプチャの後始末と，ウィンドウをすべて消す
cap.release()
cv2.destroyAllWindows()
