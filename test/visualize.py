import tensorflow as tf
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.mlab as mlab
import os
import skimage.io as io
from lib.visualize import *
from lib.utils import load_mnist_4d, save_batch_img
from test.analysis import *
from test.gan_analysis import *

MODEL_TYPENAME = "test/expr/%05d.npy"
MODEL_INDEX = 300
N_SAMPLE = 4096
DTYPE = tf.float32

def analysis_snap(model_typename, model_index):
    np_vars = np.load(model_typename % model_index)

    visualize_fc(np_vars[0])

model_list = [300, 310]
net_out = []

# set up tf
lr = tf.placeholder(DTYPE, shape=[], name="learning_rate")
X = tf.placeholder(DTYPE, [None, 784], "real_data")

trX, teX, trY, teY = load_mnist_4d("data/MNIST"); del teX, teY
trX = trX.reshape(trX.shape[0], -1) / 127.5 - 1

for idx in model_list:
    print("Model %d" % idx)
    
    # load model weight
    np_vars = np.load(MODEL_TYPENAME % idx)
    # reconstruct the model
    net_out.append(fltlist2net(X, np_vars))
    
    #visualize_fc(np_vars[0])

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

# get the output on the train data
net_out_data = sess.run(net_out, {X:trX[:N_SAMPLE]})
# see the difference of vector
diff_vector(net_out_data)