import cv2
import dlib
import numpy as np
from matplotlib import pyplot as plt
from sqlalchemy import INTEGER
import tracking
plt.gray()  # グレースケール表示に必要

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

# dlibの座標の出力形式を(x, y)のタプル形式に変換する
def part_to_coordinates(part):
    return (part.x, part.y)

def shape_to_landmark(shape):
    landmark = []
    for i in range(shape.num_parts):
        landmark.append(part_to_coordinates(shape.part(i)))
    return landmark


image_path = R"C:\Users\class\Desktop\images\i3.jpg"
img_cv2 = cv2.imread(image_path, cv2.IMREAD_COLOR)
img_RGB = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
detector = dlib.get_frontal_face_detector()
CUT_OFF = -0.1  #閾値の指定.-1を指定する事で検出する領域を意図的に増やしている
rects, scores, types = detector.run(img_cv2, 1, CUT_OFF)    #矩形, スコア, サブ検出器の結果を返す

print('------rects------')
print(rects)
print('------scores------')
print(scores)
print('------types------')
print(types)
tmp_img = img_cv2.copy()
for i, rect in enumerate(rects):    #rectsの中身をイテレート
    top, bottom, left, right = rect.top(), rect.bottom(), rect.left(), rect.right()
    cv2.rectangle(tmp_img, (left, top), (right, bottom), (0, 255, 0))

tmp_img_RGB = cv2.cvtColor(tmp_img, cv2.COLOR_BGR2RGB)

dlib_path = R"C:\Users\class\Desktop\images\shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(dlib_path)
shape = predictor(img_cv2, rects[0])

# 検出したshapeをlandmark（x,y座標のリスト）に変換
landmark = shape_to_landmark(shape)
for point in landmark:
    cv2.circle(tmp_img, point, 2, (255, 0, 255), thickness=-1)

tmp_img_RGB = cv2.cvtColor(tmp_img, cv2.COLOR_BGR2RGB)

eye_img, x_min, x_max, y_min, y_max = cut_out_eye_img(img_cv2, landmark[36:42])

# 表示確認
eye_img_copy = eye_img.copy()
landmark_local = []
for point in landmark[36:42]:
    point_local = (point[0] - x_min, point[1] - y_min)
    landmark_local.append(point_local)
    #cv2.circle(eye_img_copy, point_local, 1, (255, 0, 255), thickness=-1)  #瞳検出の座標確認用
    #plt.imshow(eye_img_copy)
    #plt.show()

#目元の切り抜き完了
tmp_img_RGB = cv2.cvtColor(eye_img_copy, cv2.COLOR_BGR2RGB)

tmp_img_GRAY = cv2.cvtColor(tmp_img_RGB, cv2.COLOR_RGB2GRAY)

aaa = np.array(landmark_local[1])
bbb = np.array(landmark_local[4])
ccc = np.array(landmark_local[2])
ddd = np.array(landmark_local[5])
aboutRadius1 = (np.linalg.norm(aaa-bbb))/2
aboutRadius2 = (np.linalg.norm(ccc-ddd))/2
aboutRadius = int((aboutRadius1+aboutRadius2)/2)
aboutRadiusa = int(aboutRadius/2)
print(aboutRadius)
circles = cv2.HoughCircles(tmp_img_GRAY, cv2.HOUGH_GRADIENT, dp=1, minDist=10, param1=100, param2=15, minRadius=(aboutRadius-aboutRadius), maxRadius=int(aboutRadius*1.15))
#circles = cv2.HoughCircles(tmp_img_GRAY, cv2.HOUGH_GRADIENT, dp=1, minDist=1, param1=170, param2=5, minRadius=(aboutRadius-aboutRadius), maxRadius=(aboutRadius))
#circlesの中身を整数値に丸めてキャスト
circles = np.uint16(np.around(circles))

#瞳周辺のランドマークの対角線の平均を出してその半分を円の半径と仮定してその円を探して描写する
a=[]
for circle in circles[0, :]:
    b=circle[2]
    a.append(b)
max_index = np.argmax(a)

# 円周を描画する
cv2.circle(tmp_img_GRAY, (circles[0][max_index][0], circles[0][max_index][1]), circles[0][max_index][2], (0, 165, 255), 1)
# 中心点を描画する
cv2.circle(tmp_img_GRAY, (circles[0][max_index][0], circles[0][max_index][1]), 2, (0, 0, 255), 1)
#for circle in circles[0, :]:
    # 円周を描画する
#    cv2.circle(tmp_img_GRAY, (circle[0], circle[1]), circle[2], (0, 165, 255), 1)
    # 中心点を描画する
#    cv2.circle(tmp_img_GRAY, (circle[0], circle[1]), 2, (0, 0, 255), 1)
plt.imshow(tmp_img_GRAY)
plt.show()