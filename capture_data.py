import cv2
import os
import pickle
import time
from utils import load_pickle_file
import yaml


with open("config.yaml") as f:
    configs = yaml.safe_load(f)
camera_configs = configs["CAMERA"]
USER = camera_configs["user"]
PASSWORD = camera_configs["password"]
IP = camera_configs["ip"]
rstp_dir = f"rstp:{USER}:{PASSWORD}@{IP}/video"
DEST_DIR = "D:\Smart parking\Data\CTIC_2023_v5"
T_SAMP = 15*60#30MIN

capturadora = cv2.VideoCapture("rtsp:admin:Hik12345@192.168.0.103/video")


time_last = time.time()
first = True
i=0
(cameraMatrix, dist) = load_pickle_file(r'camera_calibration/calibration.pkl')
while True:
    if capturadora.grab():
        ret, frame = capturadora.read()
        if not ret:
            # print("Frame lost")
            continue
        cv2.imshow('org', frame)
        h,w = frame.shape[:2]
        time_now = time.time()
        newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix, dist, (w,h), 1, (w,h))
        dst = cv2.undistort(frame, cameraMatrix, dist, None, newCameraMatrix)

        # crop the image
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]
        dst = cv2.resize(dst, (1280,720))
        cv2.imshow('Disorted', dst)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if(first or time_now-time_last>=T_SAMP):
            time_name = f"{time.asctime()}.png".replace(":", "-")
            # time_name = f"{i}.png"
            # i+=1
            file_path = os.path.join(DEST_DIR, time_name)
            cv2.imwrite(file_path, dst)
            print(f"{file_path} written")
            time_last = time_now
            first = False

capturadora.release()
cv2.destroyAllWindows()