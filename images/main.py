import copy
import cv2
import dlib
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import ImageProcessing

image_path = R"C:\Users\class\Desktop\images\i1.jpg"

image = ImageProcessing.Image(image_path)
recog = ImageProcessing.Recognition()

img_cv2 , img_gray , img_RGB = image.loading()
img_resized = image.resize(img_RGB)
image.image_display(img_resized)
rects, scores, types = recog.face_recognition(img_resized)
landmark = recog.landmark_maker(img_resized,rects)
#eye_img, x_min, x_max, y_min, y_max = recog.cut_out_eye_img(img_cv2, landmark[36:42])
#landmark_local = recog.eye_recognition(landmark,eye_img,x_min,y_min,True)  #Trueで瞳検出の座標確認(本番はFalse)

#tmp_binarizationed = image.binarization(img_gray)

#x,y,radians = recog.iris_recognition()

img_RGB_re = recog.iris(landmark , img_RGB)
H_list,S_list,V_list = recog.color(img_RGB_re)
print(H_list)
print(V_list)
#print(np.shape(H_list))

#mode = recog.HSV_mode()

#img_RGB_array = image.color_acquisition(img_RGB_re)

img_skin = recog.skin(landmark , img_resized)
H_list,S_list,V_list = recog.color(img_skin)
#print(H_list,S_list,V_list)


img_resized_copy = copy.deepcopy(img_resized)
for point in landmark:  #検出の座標確認用
    cv2.circle(img_resized_copy, point, 5, (255, 0, 255), thickness=-1)
plt.imshow(img_resized_copy)
plt.show()
image.image_display(img_skin)   #画像表示用