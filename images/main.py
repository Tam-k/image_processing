import cv2
import dlib
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import ImageProcessing

image_path = R"C:\Users\class\Desktop\images\i3.jpg"

image = ImageProcessing.Image(image_path)
img_cv2 , img_gray , img_RGB = image.loading()

recog = ImageProcessing.Recognition()
rects, scores, types = recog.face_recognition(img_RGB)
landmark = recog.landmark_maker(rects)
eye_img, x_min, x_max, y_min, y_max = recog.cut_out_eye_img(img_cv2, landmark[36:42])
recog.eye_recognition(landmark,eye_img,x_min,y_min,True)  #Trueで瞳検出の座標確認(本番はFalse)

tmp_binarizationed = image.binarization(img_gray)

x,y,radians = recog.iris_recognition()