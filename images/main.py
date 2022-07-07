import cv2
import dlib
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import ImageProcessing

image_path = R"C:\Users\class\Desktop\images\i3.jpg"

image = ImageProcessing.Image(image_path)
recog = ImageProcessing.Recognition()

img_cv2 , img_gray , img_RGB = image.loading()

rects, scores, types = recog.face_recognition(img_RGB)
landmark = recog.landmark_maker(img_cv2,rects)
#eye_img, x_min, x_max, y_min, y_max = recog.cut_out_eye_img(img_cv2, landmark[36:42])
#landmark_local = recog.eye_recognition(landmark,eye_img,x_min,y_min,True)  #Trueで瞳検出の座標確認(本番はFalse)

#tmp_binarizationed = image.binarization(img_gray)

#x,y,radians = recog.iris_recognition()

img_RGB_re = recog.iris(landmark , img_RGB)
HSV_list = recog.iris_color(img_RGB_re)
#print(np.shape(HSV_list))

mode = recog.HSV_mode(HSV_list)

#img_RGB_array = image.color_acquisition(img_RGB_re)

#image.image_display(img_HSV_re)   #画像表示用