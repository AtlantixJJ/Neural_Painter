import i2v
import os
import sys
from PIL import Image
import skimage.transform as T
import skimage.io as io
import json
import numpy as np
import csv

base_dir = sys.argv[1]

illust2vec = i2v.make_i2v_with_caffe(
     "illust2vec_tag.prototxt", "illust2vec_tag_ver200.caffemodel",
     "tag_list.json")

data = {}
array = []
filepath = os.listdir(base_dir)
filepath.sort()
csv_file = open(base_dir.replace("/", "_")+".csv", "w")

i = 0
for line in filepath:
    if not line or i > 21:
        break
    filename = base_dir + "/" + line.strip()
    try:
        img = io.imread(filename)
    except:
        print("%s open failed" % filename)
        continue
    try:
        picdict = illust2vec.estimate_plausible_tags([img], threshold=0.0)
    except:
        print("%s class failed" % filename)
        continue
    jsonObject = {}
    jsonObject["filename"] = filename
    jsonObject["result"] = picdict
    if i == 0:
        field_name = ['filename']
        field_name += [row[0] for row in picdict[0]['general']]

        csv_writer = csv.DictWriter(csv_file, fieldnames=field_name)
        csv_writer.writeheader()
    csv_dict = {}
    csv_dict['filename'] = filename
    for row in picdict[0]['general']:
        csv_dict[row[0]] = row[1]
    csv_writer.writerow(csv_dict)

    array.append(csv_dict)
    data["data"] = array

    if i % 10 == 0:
        print("running pic num "+(str)(i))

    if i % 100 == 0:
        result_json = open(sys.argv[1] + "result_json.txt", "w")
        result_json.write(json.dumps(data))
        result_json.close()

    i = i+1

labelList = ["blonde hair", "brown hair", "black hair", "blue hair", "pink hair", "purple hair", "green hair", "red hair", "silver hair", "white hair", "orange hair", "aqua hair", "grey hair",
            "long hair", "short hair",
            "twintails", "drill hair", "ponytail",
            "blush",
            "smile",
            "open mouth",
            "hat",
            "ribbon",
            "glasses",
            "blue eyes", "red eyes", "brown eyes", "green eyes", "purple eyes", "yellow eyes", "pink eyes", "aqua eyes", "black eyes", "orange eyes"]

stList = [0, 13, 15, 18, 19, 20, 21, 22, 23, 24, 34]
fiList = [12, 14, 17, 18, 19, 20, 21, 22, 23, 33, 34]
Length = 10

nowWholeArr = []
floatWholeArr = np.zeros((len(array), 35))
for cnt, jsonObject in enumerate(array):
    info =jsonObject
    nowarr = [0]

    if info['1girl'] > 0.9 and info['1boy'] < 0.1 and info['male'] < 0.1:
        passes = True
    else:
        passes = False

    #passes = True

    for catNum in range(10):
        nowmax = 0.0
        nowind = -1
        len_cat = stList[catNum+1] - stList[catNum]
        
        for labelIndex in range(stList[catNum], stList[catNum+1]):
            label = labelList[labelIndex]
            try:
                nowScore = info[label]
                floatWholeArr[cnt, labelIndex + 1] = nowScore
            except KeyError:
                print("%s not found" % label)
                nowScore = 0.0

            if len_cat > 1:
                if nowScore > nowmax:
                    nowmax = nowScore
                    nowind = labelIndex
            elif len_cat == 1:
                if nowScore > 0.5:
                    nowind = labelIndex
                else:
                    nowind = -1

        for labelIndex in range(stList[catNum], stList[catNum+1]):
            if labelIndex == nowind:
                nowarr.append(1)
            else:
                nowarr.append(0)

    if passes:
        nowarr[0] = 1
    else:
        nowarr[0] = 0
    #print(nowarr)
    nowWholeArr.append(nowarr)



np.save(sys.argv[2] + ".npy", nowWholeArr)