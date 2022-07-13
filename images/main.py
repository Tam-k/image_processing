import copy
import cv2
import dlib
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import ImageProcessing

image_path = R"C:\Users\class\Desktop\images\i10.jpg"

#別ファイルのクラスのインスタンス化
image = ImageProcessing.Image(image_path)
recog = ImageProcessing.Recognition()

#画像の読み込み、リサイズ：読込方法要検討、リサイズも場所変える？
img_cv2 , img_gray , img_RGB = image.loading()
img_resized = image.resize(img_RGB , 1000)  #縦横の最大値
#image.image_display(img_RGB)
#顔の位置を見てランドマーク作成
rects, scores, types = recog.face_recognition(img_resized)
landmark = recog.landmark_maker(img_resized,rects)

eye_img, x_min, x_max, y_min, y_max = recog.cut_out_eye_img(img_resized, landmark[36:42])

#landmark_local = recog.eye_recognition(landmark,eye_img,x_min,y_min,True)  #瞳検出の座標確認(本番はFalse) 
#tmp_binarizationed = image.binarization(img_gray)
#x,y,radians = recog.iris_recognition()

#肌色取得処理
img_skin = recog.skin(landmark , img_resized)
skin_H_list,skin_S_list,skin_V_list ,skin_HSV_array = recog.color(img_skin)
skin_S_mode_mean = recog.skin_identification(skin_H_list,skin_S_list,skin_V_list)
print("肌 S = " + str(skin_S_mode_mean))
#print(skin_S_mean)
#image.image_display(img_skin)


#白日(書き換え多分終わり)
img_resized_iris = recog.dark_eyed(landmark , img_resized)
x,y,x_2,y_2 = recog.white_eyed(landmark)
HSV_1,HSV_2 = recog.white_eye_color(img_resized,x,y,x_2,y_2)
white_eye_V = int(max(HSV_1[2],HSV_2[2]))
print(white_eye_V)
#黒目(書き換え多分終わり)
H_list,S_list,V_list ,HSV_array= recog.color(img_resized_iris)
H_list_re,S_list_re,V_list_re = recog.dark_eyed_color(H_list,S_list,V_list)
black_eye_V = int(np.mean(V_list_re))
print(black_eye_V)

contrast = recog.eye_identification(white_eye_V,black_eye_V)
print("瞳 V = "+ str(contrast))

img_iris_gray = cv2.cvtColor(eye_img, cv2.COLOR_RGB2GRAY)






#img_resized_copy = copy.deepcopy(img_resized)
""" for point in landmark:  #検出の座標確認用
    cv2.circle(img_resized_copy, point, 5, (255, 0, 255), thickness=-1)
plt.imshow(img_resized_copy)
plt.show() """
#image.image_display(eye_img)   #画像表示用