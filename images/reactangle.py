import cv2
import dlib
import numpy as np
from matplotlib import pyplot as plt
import copy

image_path = R"C:\Users\class\Desktop\images\i1.jpg"
img_cv2 = cv2.imread(image_path, cv2.IMREAD_COLOR)
img_gry = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY)
img_RGB = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
resize = 1000   #縦もしくは横の最大値にしたい数値
height, width = img_gry.shape[:2]
max_xy = max(height, width)
max_xy_copy = copy.deepcopy(max_xy)
reduced_scale = resize / max_xy_copy    #縮尺
height_re = int(height*reduced_scale)
width_re = int(width*reduced_scale)
print(height,width)
print(height_re,width_re)
img_gry = cv2.resize(img_gry , dsize=(width_re,height_re))
gray_list=[]
gray_array = np.array(img_gry)
for point in gray_array:
    for point2 in point:
        gray_list.append(point2)
print(gray_list)