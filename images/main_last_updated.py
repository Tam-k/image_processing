import copy
import cv2
import dlib
import numpy as np
#from matplotlib import pyplot as plt
#import matplotlib.patches as patches
import ImageProcessing

image_path = "C:\Users..." #処理したい画像のファイルパスを""内に入力しておく
image = ImageProcessing.Image(image_path)
recog = ImageProcessing.Recognition()

img_RGB = image.loading()
img_resized = image.resize(img_RGB , 500)  # 縦横の最大値(動作を軽くするために今回は500と設定)
    
# 顔の位置を見てランドマーク作成
rects, scores, types = recog.face_recognition(img_resized)
landmark = recog.landmark_maker(img_resized,rects)

# 肌色取得処理
img_skin = recog.skin(landmark , img_resized)
skin_H_list,skin_S_list,skin_V_list = recog.color(img_skin)
skin_S_mode_mean = recog.skin_identification(skin_S_list,skin_V_list,image_path)   # 肌結果出力用
print(f"肌 S = {str(skin_S_mode_mean)}")
img_resized_copy = copy.deepcopy(img_resized)
Mcontrast = recog.eye_contrast(img_resized_copy,landmark[36],landmark[39])  # 瞳結果出力用
#print(f"Mcontrast = {Mcontrast}")
print(str(skin_S_mode_mean), str(Mcontrast))


    #image.image_display(eye_img)   # 画像表示用
    #landmark_local = recog.eye_recognition(landmark,eye_img,x_min,y_min,True)  # 瞳検出の座標確認(本番はFalse) 
"""
    # 白日(書き換え多分終わり)
    img_resized_iris = recog.dark_eyed(landmark , img_resized)
    x,y,x_2,y_2 = recog.white_eyed(landmark)
    img_HLS = cv2.cvtColor(img_resized, cv2.COLOR_RGB2HLS)
    HLS_1,HLS_2 = recog.white_eye_color(img_HLS,x,y,x_2,y_2)
    white_eye_L = int(max(HLS_1[1],HLS_2[1]))
    
    HSV_1,HSV_2 = recog.white_eye_color(img_resized,x,y,x_2,y_2)
    white_eye_V = int(max(HSV_1[2],HSV_2[2]))
    #print(white_eye_V)
    # 黒目(書き換え多分終わり)
    H_list,S_list,V_list = recog.color(img_resized_iris)
    H_list_re,S_list_re,V_list_re = recog.dark_eyed_color(H_list,S_list,V_list)
    black_eye_L = int(np.mean(V_list_re))
    #print(black_eye_V)
    
    contrast = recog.eye_identification(white_eye_L,black_eye_L)    
    print(f"瞳 V = {str(contrast)}")
    
    cv2.circle(img_resized, (x,y), 1, (255, 0, 255), thickness=-1)  # 白目の取得位置確認用
    cv2.circle(img_resized, (x_2,y_2), 1, (255, 0, 255), thickness=-1)
    plt.imshow(img_resized)
    plt.show()
    for point in landmark:  # 検出の座標確認用
        cv2.circle(img_resized_copy, point, 5, (255, 0, 255), thickness=-1)
    plt.imshow(img_resized_copy)
    plt.show()
"""
