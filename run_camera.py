import cv2
import numpy as np
import tensorflow as tf
import time
from utils import load_spaces, crop_from_space, load_pickle_file
import queue, threading
import yaml
# bufferless VideoCapture
class VideoCapture:

  def __init__(self, name):
    self.cap = cv2.VideoCapture(name)
    self.q = queue.Queue()
    t = threading.Thread(target=self._reader)
    t.daemon = True
    t.start()

  # read frames as soon as they are available, keeping only most recent one
  def _reader(self):
    while True:
      ret, frame = self.cap.read()
      if not ret:
        break
      if not self.q.empty():
        try:
          self.q.get_nowait()   # discard previous (unprocessed) frame
        except queue.Empty:
          pass
      self.q.put(frame)

  def read(self):
    return self.q.get()

with open("config.yaml") as f:
    configs = yaml.safe_load(f)
camera_configs = configs["CAMERA"]
USER = camera_configs["user"]
PASSWORD = camera_configs["password"]
IP = camera_configs["ip"]
rstp_dir = f"rstp:{USER}:{PASSWORD}@{IP}/video"

cap= VideoCapture(rstp_dir)

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=3072)])
  except RuntimeError as e:
    print(e)
    
width= 960
height= 540

spaces = load_spaces(r"spaces/test2.txt")
model = tf.keras.models.load_model(r'models/Xception_transfer.pt')
(cameraMatrix, dist) = load_pickle_file(r'camera_calibration/calibration.pkl')
newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix, dist, (2560,1440), 1, (2560,1440))
print("Init processing")
# # used to record the time when we processed last frame
# prev_frame_time = 0
  
# # used to record the time at which we processed current frame
# new_frame_time = 0

while True:
    # if cap.cap.grab():
    # Capture frame-by-frame
        frame = cap.read()
    
    #  cv2.polylines(image, [refPt], True, (0,0,255), 2)
        # Display the resulting frame
        # cv2.imshow('Frame', frame)
        dst = cv2.undistort(frame, cameraMatrix, dist, None, newCameraMatrix)
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]
        dst = cv2.resize(dst, (1280,720), fx = 0, fy = 0,
                            interpolation = cv2.INTER_CUBIC)
        cropped_imgs = []
        for space in spaces:
            cropped_img = crop_from_space(dst, space)
            cropped_img = cv2.resize(cropped_img, (71,71))/255
            space = np.array(space).reshape(-1,1,2)
            cropped_imgs.append(cropped_img)
        cropped_imgs = np.array(cropped_imgs)
        predicts = model.predict(cropped_imgs, batch_size = 4)
        # print(predicts)
        for space, predict in zip(spaces,predicts):
            predict = predict[0]
            space = np.array(space).reshape(-1,1,2)
            if predict >= 0.5:
                # cv2.imwrite("C:\\Users\\Usuario\\Documents\\proyectos\\smart_parking\\carnet\\Occ\\"+str(i)+".jpg", cropped_img)
                color = (0,0,255)
                # print(f"{i}Occ")
            else:
                # cv2.imwrite("C:\\Users\\Usuario\\Documents\\proyectos\\smart_parking\\carnet\\Empt\\"+str(i)+".jpg", cropped_img)
                color = (0,255,0)
                # print(f"{i}Empt")
            cv2.polylines(dst, [space], True, color, 2)
            # i+=1
            # # print(cropped_img)
            # # print(cropped_img.shape)
            # # print(type(cropped_img))
            # class_predict = model.predict(img_model)[0][0]
            # # 
            # if class_predict >= 0.5:
            #     # cv2.imwrite("C:\\Users\\Usuario\\Documents\\proyectos\\smart_parking\\carnet\\Occ\\"+str(i)+".jpg", cropped_img)
            #     color = (0,0,255)
            #     # print(f"{i}Occ")
            # else:
            #     # cv2.imwrite("C:\\Users\\Usuario\\Documents\\proyectos\\smart_parking\\carnet\\Empt\\"+str(i)+".jpg", cropped_img)
            #     color = (0,255,0)
            #     # print(f"{i}Empt")
            # cv2.polylines(frame, [space], True, color, 2)
            # # i+=1
            
        dst = cv2.resize(dst, (width, height), fx = 0, fy = 0,
                            interpolation = cv2.INTER_CUBIC)
        # new_frame_time = time.time()
        # fps = 1/(new_frame_time-prev_frame_time)
        # prev_frame_time = new_frame_time
        # fps = str(int(fps))
        # cv2.putText(dst, fps, (7, 70), cv2.FONT_HERSHEY_COMPLEX, 3, (100, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow('Thresh', dst)
        # define q as the exit button
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        time.sleep(.5)
 
# release the video capture object
# cap.release()
# Closes all the windows currently opened.
cv2.destroyAllWindows()