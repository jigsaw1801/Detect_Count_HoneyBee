import cv2
import os
import json
import random

root_file = "C:/Users/Administrator/Desktop/Linh tinh/Dataset/"
path_annotations = "C:/Users/Administrator/Desktop/Linh tinh/Dataset/HoneyBee_Annotations"
save_path = "C:/Users/Administrator/Desktop/Linh tinh/Dataset/Track_image"
count = 0
# imgs = cv2.imread("C:/Users/Administrator/Desktop/Linh tinh/Dataset/HoneyBee/000001.jpg")


# colors = [random.randint(0, 255) for _ in range(3)]
# print(colors)
# imgs = cv2.rectangle(imgs, (100, 200), (150, 250), color=colors, thickness=1)
# cv2.imshow("imgs", imgs)
# cv2.waitKey(0)
colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(22)]
classes = []


def rectangle(image, x, y, w, h, cls, label=None):
    pt1 = int(x), int(y)
    pt2 = int(w), int(h)
    color = colors[int(cls)]
    cv2.rectangle(image, pt1, pt2, color=color, thickness=2)
    if label is not None:
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, fontScale=1, thickness=2)

        center = pt1[0] + 5, pt1[1] + 7 + text_size[0][1]
        pt2 = pt1[0] + 20 + text_size[0][0], pt1[1] + 20 + text_size[0][1]
        cv2.rectangle(image, pt1, pt2, color=color, thickness=-1)
        cv2.putText(image, label, center, cv2.FONT_HERSHEY_PLAIN,
                    1.5, (255, 255, 255), thickness=1)


# rectangle(imgs, 100, 200, 50, 50, label="Bee")
# cv2.imshow("imgs", imgs)
# cv2.waitKey(0)
for roots, dirs, files in os.walk(path_annotations):
    for file in files:
        path_file = os.path.join(roots, file)
        json_file = open(path_file, "r")
        data = json.load(json_file)
        path_images = root_file + data['imagePath'].strip(".")
        # print(path_images)
        img = cv2.imread(path_images)
        # cv2.imshow("img", img)
        # cv2.waitKey(10)
        obj = data['shapes']
        # print(obj)
        for i in range(len(obj)):
            cls = obj[i]['label']
            # colors = [random.randint(0, 255) for _ in range(3)]
            print(cls)
            point = obj[i]['points']

            x1 = point[0][0]
            y1 = point[0][1]
            x2 = point[1][0]
            y2 = point[1][1]

            xmin = abs(round(int(min(x1, x2))))
            xmax = abs(round(int(max(x1, x2))))
            ymin = abs(round(int(min(y1, y2))))
            ymax = abs(round(int(max(y1, y2))))
            rectangle(img, xmin, ymin, xmax, ymax, cls, label=cls)
            cv2.imshow("img", img)
        cv2.imwrite(os.path.join(save_path, "{:06d}.jpg".format(count)), img)
        count += 1
        cv2.waitKey(1)

# classes = sorted(set(classes))
# print(classes)
