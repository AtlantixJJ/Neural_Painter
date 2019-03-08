import os
import sys
from PIL import Image
import skimage.transform as T
import skimage.io as io
import json
import numpy as np
import csv

base_dir = sys.argv[1]

array = json.load(open(sys.argv[1] + "result_json.txt", "r"))[u'data']

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

    #if info['1girl'] > 0.9 and info['1boy'] < 0.1 and info['male'] < 0.1:
    #    passes = True
    #else:
    #    passes = False

    passes = True

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
        nowarr[0] = floatWholeArr[cnt, 0] = 1
    else:
        nowarr[0] = floatWholeArr[cnt, 0] = 0
    #print(nowarr)
    nowWholeArr.append(nowarr)
    floatWholeArr.append(floatarr)

for i in range(len(stList)-1):
    # skip the switch variable
    # TODO: change switch variable to two-alternative variable
    # The origin paper assumes dense prob to 0
    if stList[i+1] - stList[i] <= 1:
        floatWholeArr[j, stList[i]] *= floatWholeArr[j, stList[i]]
    for j in range(len(array)):
        floatWholeArr[j, stList[i]: stList[i+1]] /= floatWholeArr[j, stList[i]: stList[i+1]].sum()

np.save(sys.argv[2] + ".npy", nowWholeArr)
np.save(sys.argv[2] + "float.npy", floatWholeArr)