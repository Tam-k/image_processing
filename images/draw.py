import cv2
import numpy as np
import copy
image_path= R"C:\Users\class\Desktop\img_RGB.jpg"
img_cv2 = cv2.imread(image_path, cv2.IMREAD_COLOR)
img_HSV = cv2.cvtColor(img_cv2,cv2.COLOR_BGR2HSV)
img_BGR_array = np.array(img_cv2)
img_HSV_array=copy.deepcopy(img_BGR_array)
#print(img_HSV_array)
i=0
for point1 in img_BGR_array:
    j=0
    for point2 in point1:
        H=int(0)
        max = np.argmax(point2)
        min = np.argmin(point2)
        G=point2[0]
        B=point2[1]
        R=point2[2]
        if (G==B==R):
            H=0
        else:
            diff=point2[max] - point2[min]
            if max==0:
                H = int((60*((G-R)/diff))/2)
            elif max==1:
                H = int((60*((R-G)/diff) +120)/2)
            elif max==2:
                H = int((60*((G-B)/diff) +240)/2)
        if H < 0 :
            H += 360
        if point2[max] != 0:
            S = diff/ point2[max]
        else:
            S = 0
        V = point2[max]
        img_HSV_array[i][j][0]=H
        img_HSV_array[i][j][1]=S
        img_HSV_array[i][j][2]=V
        j+=1
    i+=1
print(img_HSV_array)
print("/////////////////////////////")
print(np.array(img_HSV))
print(i,j)
cv2.imwrite("i3eye.jpg" , img_HSV_array)



"""
HSVの大体の色
330-360,0-30 赤
30-90 黄
90-150 緑
150-210 水
210-270 青
270-330 紫
"""