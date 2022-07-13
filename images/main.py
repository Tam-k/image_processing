import copy
import cv2
import dlib
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import ImageProcessing

image_path = R"C:\Users\class\Desktop\images\i1.jpg"

#別ファイルのクラスのインスタンス化
image = ImageProcessing.Image(image_path)
recog = ImageProcessing.Recognition()

#画像の読み込み、リサイズ：読込方法要検討、リサイズも場所変える？
img_cv2 , img_gray , img_RGB = image.loading()
img_resized = image.resize(img_RGB)

#顔の位置を見てランドマーク作成
rects, scores, types = recog.face_recognition(img_resized)
landmark = recog.landmark_maker(img_resized,rects)
""" 
eye_img, x_min, x_max, y_min, y_max = recog.cut_out_eye_img(img_resized, landmark[36:42])
landmark_local = recog.eye_recognition(landmark,eye_img,x_min,y_min,True)  #瞳検出の座標確認(本番はFalse)
#tmp_binarizationed = image.binarization(img_gray)
 """
#x,y,radians = recog.iris_recognition()

#肌色取得処理
img_skin = recog.skin(landmark , img_resized)
skin_H_list,skin_S_list,skin_V_list ,skin_HSV_array = recog.color(img_skin)




#白日(書き換えほとんど終わり)
img_resized_iris = recog.dark_eyed(landmark , img_resized)
x,y,x_2,y_2 = recog.white_eyed(landmark)
HSV_1,HSV_2 = recog.white_eye_color(img_resized,x,y,x_2,y_2)
""" cv2.circle(img_resized, (x,y), 1, (255, 0, 255), thickness=-1)
cv2.circle(img_resized, (x_2,y_2), 1, (255, 0, 255), thickness=-1)
plt.imshow(img_resized)
plt.show()
image.save(img_resized)"""

#黒目(書き換え途中)
H_list,S_list,V_list ,HSV_array= recog.color(img_resized_iris)
H_list_re=copy.deepcopy(H_list)
S_list_re=copy.deepcopy(S_list)
V_list_re=copy.deepcopy(V_list)
H_list_re,S_list_re,V_list_re = image.V_cutter(H_list_re,S_list_re,V_list_re)
#image.image_display(img_resized_iris)
mode = recog.dark_eyed_color(img_resized_iris,H_list,S_list,V_list)
#print(mode)

#img_RGB_array = image.color_acquisition(img_RGB_re)


#img_resized_copy = copy.deepcopy(img_resized)
""" for point in landmark:  #検出の座標確認用
    cv2.circle(img_resized_copy, point, 5, (255, 0, 255), thickness=-1)
plt.imshow(img_resized_copy)
plt.show() """
#image.image_display(img_skin)   #画像表示用