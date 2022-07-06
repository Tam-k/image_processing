#画像の扱い : img=元画像 , tmp=編集後

import cv2
import dlib
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as patches
class Image:
    def __init__(self,image_path):
        self.image_path=image_path
    
    def loading(self):
        img_cv2 = cv2.imread(self.image_path, cv2.IMREAD_COLOR)
        img_gry = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY)
        img_RGB = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
        return (img_cv2,img_gry,img_RGB)
    
    def binarization(img_gray):
        #大津の二値化
        #ret, tmp_binarizationed = cv2.threshold(img_gray, 200, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        #tmp_binarizationed = cv2.bitwise_not(img_gray) #白黒反転
        #Canny法の二値化
        tmp_binarizationed = cv2.Canny(img_gray,160,160)
        #適応的閾値処理
        #tmp_binarizationed = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 39, 2)
        return tmp_binarizationed
    
    
    
class Recognition:
    def __init__(self) -> None:
        pass
    
    # dlibの座標の出力形式を(x, y)のタプル形式に変換する
    def part_to_coordinates(part):
        return (part.x, part.y)
    def shape_to_landmark(shape):
        landmark = []
        for i in range(shape.num_parts):
            landmark.append(Recognition.part_to_coordinates(shape.part(i)))
        return landmark
    
    def face_recognition(self,img_RGB):
        detector = dlib.get_frontal_face_detector()
        CUT_OFF = -0.1  #閾値の指定.-1を指定する事で検出する領域を意図的に増やしている
        rects, scores, types = detector.run(img_RGB, 1, CUT_OFF)    #矩形, スコア, サブ検出器の結果を返す
        return rects, scores, types
    
    def landmark_maker(img_cv2,rects):
        tmp_img = img_cv2.copy()
        for i, rect in enumerate(rects):    #rectsの中身をイテレート
            top, bottom, left, right = rect.top(), rect.bottom(), rect.left(), rect.right()
            cv2.rectangle(tmp_img, (left, top), (right, bottom), (0, 255, 0))

        tmp_img_RGB = cv2.cvtColor(tmp_img, cv2.COLOR_BGR2RGB)

        dlib_path = R"C:\Users\class\Desktop\images\shape_predictor_68_face_landmarks.dat"
        predictor = dlib.shape_predictor(dlib_path)
        shape = predictor(img_cv2, rects[0])

        # 検出したshapeをlandmark（x,y座標のリスト）に変換
        landmark = Recognition.shape_to_landmark(shape)
        for point in landmark:
            cv2.circle(tmp_img, point, 2, (255, 0, 255), thickness=-1)
        return landmark
    
    def cut_out_eye_img(img_cv2, eye_points):
        height, width = img_cv2.shape[:2]
        x_list = []
        y_list = []
        for point in eye_points:
            x_list.append(point[0])
            y_list.append(point[1])
        x_min = max(min(x_list) - 3, 0)
        x_max = min(max(x_list) + 4, width)
        y_min = max(min(y_list) - 3, 0)
        y_max = min(max(y_list) + 4, height)
        eye_img = img_cv2[y_min : y_max, x_min : x_max]
        return eye_img, x_min, x_max, y_min, y_max
    
    def eye_recognition(landmark,eye_img,x_min,y_min,boo):
        # 表示確認(右目のみ)
        eye_img_copy = eye_img.copy()
        landmark_local = []
        for point in landmark[36:42]:
            point_local = (point[0] - x_min, point[1] - y_min)
            landmark_local.append(point_local)
            if boo:
                cv2.circle(eye_img_copy, point_local, 1, (255, 0, 255), thickness=-1)  #瞳検出の座標確認用
                plt.imshow(eye_img_copy)
                plt.show()
        return landmark_local
                
    def iris_recognition(landmark_local,tmp_binarizationed):  #瞳周辺のランドマークの対角線の平均を出してその半分を円の半径と仮定してその円を探して描写する
        aaa = np.array(landmark_local[1])
        bbb = np.array(landmark_local[4])
        ccc = np.array(landmark_local[2])
        ddd = np.array(landmark_local[5])
        radius1 = (np.linalg.norm(aaa-bbb))/2
        radius2 = (np.linalg.norm(ccc-ddd))/2
        radius = int((radius1+radius2)/2)
        circles = cv2.HoughCircles(tmp_binarizationed,cv2.HOUGH_GRADIENT,dp=1,minDist=1,param1=150,param2=20,minRadius=int(radius*0.6), maxRadius=int(radius*1.3))    #画質が荒い時はparam2を下げる(5)にするほうが検知がしやすそう
        circles = np.uint16(np.around(circles)) #circlesの中身を整数値に丸めてキャスト
        a=[]
        for circle in circles[0, :]:
            b=circle[2]
            a.append(b)
        max_index = np.argmax(a)
        return circles[0][max_index][0], circles[0][max_index][1], circles[0][max_index][2]