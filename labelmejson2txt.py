import os
import json
from PIL import Image


def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


path = "C:\\Users\\Administrator\\Desktop\\Linh tinh\\Dataset\\efficientdet\\Bee\\Annotation_HoneyBee1\\"
out_path = "C:\\Users\\Administrator\\Desktop\\Linh tinh\\Dataset\\efficientdet\\Bee\\labels\\"


# wd = os.getcwd()
# list_file = open('%s_list.txt' % (wd), 'w')
# print(list_file)
json_name_list = []
for file in os.listdir(path):
    if file.endswith(".json"):
        json_name_list.append(file)

for json_name in json_name_list:
    print(json_name)
    txt_name = json_name.rstrip(".json") + ".txt"
    print(txt_name)
    txt_path = path + json_name
    print("Input:" + txt_path)
    txt_file = open(txt_path, "r")

    txt_outpath = out_path + txt_name
    print("Output:" + txt_outpath)
    txt_outfile = open(txt_outpath, "w")
    data = json.load(txt_file)
    print(data)
    obj = data['shapes']
    print(obj)
    for i in range(0, len(obj)):
        point = obj[i]['points']
        x1 = point[0][0]
        y1 = point[0][1]
        x2 = point[1][0]
        y2 = point[1][1]
        xmin = min(x1, x2)
        xmax = max(x1, x2)
        ymin = min(y1, y2)
        ymax = max(y1, y2)
        w = 1280
        h = 720
        print(xmin, xmax, ymin, ymax)
        b = (xmin, xmax, ymin, ymax)
        bb = convert((w, h), b)
        print(bb)
        txt_outfile.write(str(0) + " " + " ".join([str(a) for a in bb]) + '\n')

    # point = obj['points']
    # print(point)
    # lines = txt_file.read().split('\n')
    # for idx, line in enumerate(lines):
    #     if ("label" in line ):
    #         x1 = float(lines[idx + 3].rstrip(','))
    #         y1 = float(lines[idx + 4])
    #
    #         x2 = float(lines[idx + 7].rstrip(','))
    #         y2 = float(lines[idx + 8])
    #         # cls = line[16:17]
    #         #
    #         #             # in case when labelling, points are not in the right order
    #         xmin = min(x1, x2)
    #         xmax = max(x1, x2)
    #         ymin = min(y1, y2)
    #         ymax = max(y1, y2)
    #         # img_path = str('%s/dataset/%s.jpg' % (wd, os.path.splitext(json_name)[0]))
    #
    #         # im = Image.open(img_path)
    #         w = 416
    #         h = 416
    #
    #
    #         print(xmin, xmax, ymin, ymax)
    #         b = (xmin, xmax, ymin, ymax)
    #         bb = convert((w, h), b)
    #         print(bb)
    #         txt_outfile.write(str(1) + " " + " ".join([str(a) for a in bb]) + '\n')

#     os.rename(txt_path, json_backup + json_name)  # move json file to backup folder
#
#     """ Save those images with bb into list"""
#     if (txt_file.read().count("label") != 0):
#         list_file.write('%s/dataset/%s.jpg\n' % (wd, os.path.splitext(txt_name)[0]))
#
# list_file.close()
