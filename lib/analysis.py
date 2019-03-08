"""
Analysis utils migrated from GAN analysis.
"""

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.mlab as mlab
import os
import skimage.io as io

def read_csv(fname):
    l = []
    with open(fname, 'rb') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')

        for i,row in enumerate(spamreader):
            if i == 0:
                continue
            l.append(float(row[0].split(",")[2]))
    return l

def hist(f, name, range=None):
    plt.hist(f, bins=100, range=range, normed=True)
    plt.savefig(name)
    plt.close()

def add_normal(mean, std):
    x = np.linspace(mean-4*std, mean+4*std, 100)
    #x = np.linspace(mean - 3*std, mean + 3*std, 100)
    plt.plot(x,mlab.normpdf(x, mean, std))

def plot(f, name):
    plt.plot(f)
    plt.savefig(name)
    plt.close()

def histn(f, name, mean, std):
    plt.hist(f, bins=32,range=(mean-4*std, mean+4*std), normed=True)
    add_normal(mean, std)
    plt.savefig(name)
    plt.close()

def read_imgs(dir):
    imgs = []
    for f in os.listdir(dir):
        if f.find("png") == -1:
            continue

        fpath = dir + "/" + f
        imgs.append(io.imread(fpath))
    return np.array(imgs)

def edge(img):
    grad_x = np.abs(img[1:, :] - img[:-1, :])
    grad_y = np.abs(img[:, 1:] - img[:, :-1])
    return grad_x[:, 1:] + grad_y[1:, :]

def abs_loss(img1, img2):
    return np.sum(np.abs(img1 - img2))/16.

def get_abs_loss(imgs, bl=1):
    t = []
    for j in range(1, bl+1, 1):
        l = []
        for i in range(1, imgs.shape[0]):
            if i-j>=0:
                l.append(abs_loss(imgs[i-j], imgs[i]))
            else:
                l.append(784.)
        t.append(l)
    t = np.array(t)
    print(t.shape)
    return t.min(0)