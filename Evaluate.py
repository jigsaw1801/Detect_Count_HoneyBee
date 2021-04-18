import json
import os
import cv2
import numpy as np

path_predict = "C:/Users/Administrator/Desktop/Linh tinh/Dataset/HoneyBee1_annotations/"
path_ground_truth = "C:/Users/Administrator/Desktop/Linh tinh/Dataset/yolov5/ground_truth/"
path_images = "C:/Users/Administrator/Desktop/Linh tinh/Dataset/yolov5/exp15/"
path_evaluate = "C:/Users/Administrator/Desktop/Linh tinh/Dataset/yolov5/Evaluate_video/"
gt_boxes = []
pre_boxes = []
TP = 0
FP = 0
FN = 0
sum_FP = 0
sum_TP = 0
sum_FN = 0
count = 202


def get_box(obj, boxes_list):
    for i in range(len(obj)):
        bbox = obj[i]['points']
        x1 = bbox[0][0]
        y1 = bbox[0][1]
        x2 = bbox[1][0]
        y2 = bbox[1][1]
        xmin = min(x1, x2)
        xmax = max(x1, x2)
        ymin = min(y1, y2)
        ymax = max(y1, y2)
        box = (xmin, ymax, xmax, ymin)
        boxes_list.append(box)


def bb_intersection_over_union(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = min(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = max(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yA - yB + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[1] - boxA[3] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[1] - boxB[3] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def get_path(path, lists):
    for file in os.listdir(path):
        lists.append(file)


def calc_conditions(gt_boxes, pred_boxes, iou_thresh):
    gt_class_ids_ = np.zeros(len(gt_boxes))
    pred_class_ids_ = np.zeros(len(pred_boxes))

    TP, FP, FN = 0, 0, 0
    for i, gt_box in enumerate(gt_boxes):
        iou = []
        for j, pre_box in enumerate(pre_boxes):
            now_iou = bb_intersection_over_union(gt_box, pre_box)
            if now_iou >= iou_thresh:
                iou.append(now_iou)
                gt_class_ids_[i] = 1
                pred_class_ids_[j] = 1
        if len(iou) > 0:
            TP += 1
            FP += len(iou) - 1
    FN += np.count_nonzero(np.array(gt_class_ids_) == 0)
    FP += np.count_nonzero(np.array(pred_class_ids_) == 0)
    false_alarm_rate = round(FP * 100 / (FP + TP), 2)
    miss_rate = round(FN * 100 / (FN + TP), 2)
    return TP, FP, FN, false_alarm_rate, miss_rate


if __name__ == "__main__":
    predict_name_list = []
    ground_name_list = []
    images_name_list = []
    sum_FP = 0
    sum_TP = 0
    sum_FN = 0
    get_path(path_predict, predict_name_list)
    get_path(path_ground_truth, ground_name_list)
    get_path(path_images, images_name_list)
    print(len(images_name_list))
    for i in range(len(predict_name_list)):
        path_predict_file = path_predict + predict_name_list[i]
        path_ground_file = path_ground_truth + ground_name_list[i]
        gt_file = open(path_ground_file, "r")
        pre_file = open(path_predict_file, "r")
        gt_json = json.load(gt_file)
        pre_json = json.load(pre_file)
        gt_file.close()
        pre_file.close()
        gt_obj = gt_json['shapes']
        pre_obj = pre_json['shapes']
        get_box(gt_obj, gt_boxes)
        get_box(pre_obj, pre_boxes)
        TP, FP, FN, false_alarm_rate, miss_rate = calc_conditions(gt_boxes, pre_boxes, iou_thresh=0.5)
        print("tp :", TP)
        print("fp :", FP)
        print("fn :", FN)

        sum_TP += TP
        sum_FP += FP
        sum_FN += FN
        image_file = path_images + images_name_list[i]
        img = cv2.imread(image_file)
        # label1 = f"Accuracy : {Accuracy}%"
        label1 = f"False_alarm_rate : {false_alarm_rate}%"
        label2 = f"Miss_rate : {miss_rate}%"
        img1 = cv2.putText(img,
                           label1,
                           org=(900, 100),
                           fontFace=cv2.FONT_HERSHEY_PLAIN,
                           fontScale=1.75,
                           color=(255, 255, 255),
                           thickness=2
                           )
        img3 = cv2.putText(img1,
                           label2,
                           org=(900, 150),
                           fontFace=cv2.FONT_HERSHEY_PLAIN,
                           fontScale=1.75,
                           color=(255, 255, 255),
                           thickness=2
                           )
        # cv2.imshow("img0", img3)
        # cv2.waitKey(100)
        # cv2.imwrite(os.path.join(path_evaluate, "{:06d}.jpg".format(count)), img3)
        # count += 1
        pre_boxes = []
        gt_boxes = []
        print(path_predict_file)
# sum_Accuracy = round(sum_TP * 100 / (sum_TP + sum_FP + sum_FN), 2)
sum_false_alarm_rate = round(sum_FP * 100 / (sum_TP + sum_FP), 2)
sum_Miss_rate = (round(sum_FN * 100 / (sum_FN + sum_TP), 2))
# print(sum_Accuracy)
print("false_alarm_rate", sum_false_alarm_rate)
print("miss_rate", sum_Miss_rate)
