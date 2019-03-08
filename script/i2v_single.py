"""
Classify single image using illustration2vec.
Requirement:
1. module i2v
2. illust2vec_tag.prototxt
3. illust2vec_tag_ver200.caffemodel
"""

import i2v
from PIL import Image
import numpy as np
import skimage.transform as T
import skimage.io as io

illust2vec = i2v.make_i2v_with_caffe(
        "illust2vec_tag.prototxt", "illust2vec_tag_ver200.caffemodel",
        "tag_list.json")
img = io.imread("faces_safebooru/1.jpg")

predict = illust2vec.estimate_plausible_tags([img], threshold=0.01)
