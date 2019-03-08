"""
Two stage GAN training pipeline. (Currently this is second stage)
GAN takes in real data and sketch & mask pair, try to modify it!
"""
import matplotlib
matplotlib.use("agg")
import sys
sys.path.insert(0, ".")
import tensorflow as tf

import time
import pprint
import numpy as np
from scipy import misc
import os

# model
import model
import config as cfg
import loss
import trainer
import skimage
from lib import utils, dataloader, ops

tf.app.flags.DEFINE_integer("gpu", 6, "which gpu to use")
FLAGS = tf.app.flags.FLAGS
TFLAGS = {}

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.gpu)

size = 128
x_real = tf.placeholder(tf.float32, [None, size, size, 3], name="x_real")

img = skimage.io.imread("x_fake_sample.png") / 127.5 - 1
print(img.shape)
img = np.expand_dims(skimage.transform.resize(img, (size, size)), 0)

net = model.classifier.DilatedVGG16("lib/tensorflowvgg/vgg16.npy")
net.build(x_real)
sess = tf.InteractiveSession()
a = net.conv3_3.eval({x_real:img})