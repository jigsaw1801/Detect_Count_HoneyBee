import cv2
import json
import os
import matplotlib.pyplot as plt

path_annotations = 'C:/Users/Administrator/Desktop/Linh tinh/Dataset/yolov5/Bee/Annotation_HoneyBee/'
path_crop = "C:/Users/Administrator/Desktop/Linh tinh/Dataset/crop_images/"
Bee = []
count = 0
number_box = 0
path = path_annotations.split("Annotation_HoneyBee1/")[0]
for roots, dirs, files in os.walk(path_annotations):
    for file in files:
        path_file = roots + file
        with open(path_file) as f:
            data = json.load(f)
            img_path = path + data['imagePath'].strip(".")
            obj = data['shapes']
            number_box = number_box + len(obj)
            for i in range(len(obj)):
                point = obj[i]['points']
                x1 = point[0][0]
                y1 = point[0][1]
                x2 = point[1][0]
                y2 = point[1][1]

                xmin = abs(round(int(min(x1, x2))))
                xmax = abs(round(int(max(x1, x2))))
                ymin = abs(round(int(min(y1, y2))))
                ymax = abs(round(int(max(y1, y2))))
                print(img_path)

                img = cv2.imread(img_path)
                cv2.imshow("img", img)
                crop_img = img[ymin:ymax, xmin:xmax]
                cv2.imshow("crop", crop_img)
                cv2.imwrite(os.path.join(path_crop, "{:06d}.jpg".format(count)), crop_img)
                count = count + 1
print(number_box)

# img = cv2.imread("C:/Users/Administrator/Desktop/Linh tinh/Dataset/efficientdet/Bee/HoneyBee1/000000.jpg")
# with open("C:/Users/Administrator/Desktop/Linh tinh/Dataset/efficientdet/Bee/Annotation_HoneyBee1/000000.json") as f:
#     data = json.load(f)
#     obj = data['shapes']
#     for i in range(len(obj)):
#         point = obj[i]['points']
#         x1 = point[0][0]
#         y1 = point[0][1]
#         x2 = point[1][0]
#         y2 = point[1][1]
#
#         xmin = int(min(x1, x2))
#         xmax = int(max(x1, x2))
#         ymin = int(min(y1, y2))
#         ymax = int(max(y1, y2))
#
#         cv2.rectangle(img, (int(xmin), int(ymax)), (int(xmax), int(ymin)), color=(0, 255, 0), thickness=2)
#         label = obj[i]['label']
#         print(label)
#
#         ((label_w, label_h), _) = cv2.getTextSize(
#             label,
#             fontFace=cv2.FONT_HERSHEY_PLAIN,
#             fontScale=1.75,
#             thickness=2
#         )
#         cv2.rectangle(
#             img,
#             (int(xmin), int(ymin)),
#             (int(xmin + label_w), int(ymin - 1.5 * label_h)),
#             color=(0, 255, 0),
#             thickness=cv2.FILLED
#         )
#         cv2.putText(
#             img,
#             label,
#             org=(int(xmin), int(ymin)),
#             fontFace=cv2.FONT_HERSHEY_PLAIN,
#             fontScale=1.75,
#             color=(255, 255, 255),
#             thickness=2
#
#
#         )
#
#     cv2.imshow("img", img)
#     cv2.waitKey(0)
