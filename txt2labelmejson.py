import json
import cv2
import os

# (x1 + x2)/(2*1920) = a
# (y1 + y2)/(2*1080) = b
# (x1 - x2)/1920 = c
# (y1 - y2)/1080 = d
null = None
a = []
i = 0
count = 202
path = "C:/Users/Administrator/Desktop/Linh tinh/Dataset/exp14/labels/"
json_path = "C:/Users/Administrator/Desktop/Linh tinh/Dataset/HoneyBee1_annotations/"
# content = {"version": "4.5.6", "flags": {}}
# for i in range(236):
#     file_path = json_path + "{:06d}.json".format(i)
#     file = open(file_path, 'w')


json_name_list = []
for file in os.listdir(json_path):
    if file.endswith(".json"):
        json_name_list.append(file)


for root, dirs, files in os.walk(path):
    for file in files:
        txt_path = os.path.join(root, file)
        with open(txt_path) as f:
            for line in f:
                value = line.split(" ")
                v0 = float(value[0])
                v1 = float(value[1])
                v2 = float(value[2])
                v3 = float(value[3])
                v4 = float(value[4])
                v = (v0, v1, v2, v3, v4)
                w, h = 1280, 720
                x1 = (v[1] * w * 2 + v[3] * w) / 2
                x2 = (v[1] * w * 2 - v[3] * w) / 2
                y1 = (v[2] * h * 2 + v[4] * h) / 2
                y2 = (v[2] * h * 2 - v[4] * h) / 2
                xmin = min(x1, x2)
                xmax = max(x1, x2)
                ymin = min(y1, y2)
                ymax = max(y1, y2)
                print(xmax)
                # point1 = (xmin, ymax)
                # point2 = (xmax, ymin)
                dict = {
                    "label": "HoneyBee",
                    "points": [
                        [
                            xmin,
                            ymax
                        ],
                        [
                            xmax,
                            ymin
                        ]
                    ],
                    "group_id": null,
                    "shape_type": "rectangle",
                    "flags": {}
                }
                a.append(dict)

        path_json = json_path + json_name_list[i]
        json_file = open(path_json, 'r')
        data = json.load(json_file)
        json_file.close()
        f.close()
        # data["version"] = "4.5.6"
        # data["flags"] = {}
        data['shapes'] = a
        data["imagePath"] = "..\\HoneyBee1\\{:06d}.jpg".format(count)
        data["imageData"] = null
        data["imageHeight"] = 720
        data["imageWidth"] = 1280
        json_file = open(path_json, "w")
        json.dump(data, json_file)
        json_file.close()
        a = []
        i += 1
        count = count + 1
