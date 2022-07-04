import sys
import cv2
import numpy as np
import os

# .xmlはLib/site-packages/cv2/data/にある
face_cascade_path = R'C:\Users\class\anaconda3\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml'
eye_cascade_path = R'C:\Users\class\anaconda3\Lib\site-packages\cv2\data\haarcascade_eye.xml'

face_cascade = cv2.CascadeClassifier(face_cascade_path)
eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
image_path = R"C:\Users\class\Desktop\images\i1.jpg"  #後々、取得する画像にパスを変更する
src = cv2.imread(image_path)
if os.path.exists(image_path):
    src_gray = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
    cv2.imwrite('src_gray.jpg', src)
else:
    sys.exit("\nImageException : 画像を読み込めませんでした。\nプログラムを終了します。")

faces = face_cascade.detectMultiScale(src_gray)
src_hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
cv2.imwrite('src_hsv.jpg', src_hsv)
i=0
for x, y, w, h in faces:
    cv2.rectangle(src, (x, y), (x + w, y + h), (255, 0, 0), 1)
    face = src[y: y + h, x: x + w]
    face_gray = src_gray[y: y + h, x: x + w]
    eyes = eye_cascade.detectMultiScale(face_gray)
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(face, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 1)
        xmin = ex
        xmax = ex+ew
        ymin = ey
        ymax = ey+eh
        print(ymin,ymax,xmin,xmax)
        detframe = face_gray[ymin:ymax,xmin:xmax]
        detframe = cv2.bilateralFilter(detframe,3,125,125)
        img = cv2.bitwise_not(detframe)
        ret, image = cv2.threshold(img, 50 , 255 , cv2.THRESH_BINARY)
        if i==0:
            cv2.imwrite('aaaaaa.jpg', detframe)
            cv2.imwrite('aaaaab.jpg', image)
            detframe = cv2.Canny(detframe,100,100)
            cv2.imwrite('bbbbbb.jpg', detframe)
        elif i==1:
            cv2.imwrite('cccccc.jpg', detframe)
            cv2.imwrite('aaaaac.jpg', image)
            detframe = cv2.Canny(detframe,200,200)
            cv2.imwrite('dddddd.jpg', detframe)
        else:
            sys.exit("\nImageException : 画像処理にエラー発生。\nプログラムを終了します。")
        i+=1
cv2.imwrite('face_rectangle.jpg', src)
#  cv2.imwrite('img.jpg', img)

#  注意！！
#  ・顔全体が映っていないと瞳の検出が出来ない(おそらく顔の中に瞳がある前提があるため)

#  ・画像の真ん中から近い部分の白を読み取って座標を取得する