import cv2
import numpy as np
import os
import pandas as pd
import shutil

count = 0
path_coordinates = "C:/Users/Administrator/Desktop/Linh tinh/Dataset/coordinate/Crawling/"
Bee = cv2.imread("Bee.jpg")
Bee = cv2.resize(Bee, (20, 18))
Bee_flip = cv2.flip(Bee, 1)
img = np.zeros([720, 1280, 3], dtype=np.uint8)
img.fill(255)
img = cv2.line(img, (192, 231), (829, 232), color=(255, 0, 0), thickness=5)
img = cv2.line(img, (829, 233), (1055, 240), color=(255, 0, 0), thickness=5)
img = cv2.line(img, (1055, 240), (1280, 250), color=(255, 0, 0), thickness=5)
coor = []
name_list = []

for file in os.listdir(path_coordinates):
    if file.endswith(".txt"):
        name_list.append(file)

for name in name_list:
    path_file = path_coordinates + name
    a = pd.read_csv(path_file, sep='\n', header=None, engine='python')
    for i in range(len(a)):
        coordinate = a[0][i]
        b = coordinate.split(" ")
        coor.append(b)
        coor[i][0] = int(coor[i][0])
        coor[i][1] = int(coor[i][1])
    print(coor)

    for j in range(len(coor) - 2):
        # x = coor[j][0]
        # y = coor[j][1]
        h, w = Bee.shape[:2]
        b, a = coor[j][0], coor[j][1]
        y, x = a - int(h / 2), b - int(w / 2)
        img = cv2.line(img, (coor[j][0], coor[j][1]), (coor[j + 1][0], coor[j + 1][1]), color=(0, 255, 0), thickness=5)
        img1 = img.copy()
        img[y:y + h, x:x + w] = Bee
        # cv2.imwrite(os.path.join(path_coordinates, "{:06d}.jpg".format(count)), img)
        # count += 1
        cv2.imshow("img", img)
        cv2.waitKey(500)
        img = img1

    for k in range(len(coor) - 2, len(coor) - 1):
        h, w = Bee.shape[:2]
        b, a = coor[k][0], coor[k][1]
        y, x = a - int(h / 2), b - int(w / 2)
        img = cv2.arrowedLine(img, (coor[k][0], coor[k][1]), (coor[k + 1][0], coor[k + 1][1]), color=(2, 255, 0),
                              thickness=5, tipLength=0.2)
        img1 = img.copy()
        img[y:y + h, x:x + w] = Bee
        cv2.imwrite(os.path.join(path_coordinates, "{:06d}.jpg".format(count)), img)
        count += 1
        cv2.imshow("img", img)
        cv2.waitKey(500)
        img = img1

    coor = []
# name_file = path_coordinates + "img.jpg"
# # if os.path.exists(name_file):
# #     shutil.rmtree(name_file)
# cv2.imwrite(name_file, img)
