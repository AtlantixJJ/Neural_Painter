import csv
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.mlab as mlab
import os
import skimage.io as io

LEN = 1
title = "dragan_"
fname_disc = "run_simple_dragan_mnist1-tag-Naive_Discriminator_Loss_Fake__raw_.csv"
fname_gen = "run_simple_dragan_mnist1-tag-Total_Generator_Loss.csv"

#title = "naive_"
#fname_disc = "run_simple_naive_mnist1-tag-Naive_Discriminator_Loss_Fake__raw_.csv"
#fname_gen = "run_simple_naive_mnist1-tag-Total_Generator_Loss.csv"

def read_csv(fname):
    l = []
    with open(fname, 'rb') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')

        for i,row in enumerate(spamreader):
            if i == 0:
                continue
            l.append(float(row[0].split(",")[2]))
    return l

def interleave(l1, l2):
    a = np.zeros((2 * l1.shape[0],))
    a[::2] = l1
    a[1::2] = l2
    return a

def clip_len(l1, l2):
    min_len = min(l1.shape[0], l1.shape[0])
    l1_ = l1[:min_len]; l2_ = l2[:min_len]
    return l1_, l2_

def hist(f, name, range=None):
    plt.hist(f, bins=100, range=range, normed=True)
    plt.savefig(name)
    plt.close()

def std_mean(x):
    return np.std(x), np.mean(x)

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

def main():
    d_loss_fake = np.array(read_csv(fname_disc))
    g_loss      = np.array(read_csv(fname_gen))
    d_loss_fake, g_loss = clip_len(d_loss_fake, g_loss)
    print(len(d_loss_fake), len(g_loss))
    # Tn is total accumulative
    Tn          = interleave(d_loss_fake, g_loss)
    # Zn is total increase
    Zn          = d_loss_fake[1:] - d_loss_fake[:-1]
    # Yn is gen increase
    Yn          = g_loss[:-1] - d_loss_fake[:-1]
    # Xn is disc increase
    Xn          = d_loss_fake[1:] - g_loss[:-1]

    Zn_std, Zn_mean = std_mean(Zn)
    Xn_std, Xn_mean = std_mean(Xn)
    Yn_std, Yn_mean = std_mean(Yn)

    assert((Xn + Yn == Zn).all())

    plot(d_loss_fake,   title+"d_loss_fake.png")
    plot(g_loss,        title+"g_loss.png")
    plot(Tn,            title+"Tn.png")
    plot(Xn,            title+"Xn.png")
    plot(Yn,            title+"Yn.png")
    plot(Zn,            title+"Zn.png")
    histn(Xn,           title+"Xn_hist.png", Xn_mean, Xn_std)
    histn(Yn,           title+"Yn_hist.png", Yn_mean, Yn_std)
    histn(Zn,           title+"Zn_hist.png", Zn_mean, Zn_std)

    gen_imgs = read_imgs("expr/simple_dragan_mnist1")
    #means = gen_imgs.reshape(gen_imgs.shape[0], -1).mean(1).reshape(-1, 1, 1)
    #std = gen_imgs.reshape(gen_imgs.shape[0], -1).std(1).reshape(-1, 1, 1)
    #print(means.shape, std.shape)
    #gen_imgs = (gen_imgs - means)/std
    gen_imgs[gen_imgs<127]=0
    gen_imgs[gen_imgs>=127]=1
    #avg_img = gen_imgs[20:].mean(0)
    #io.imsave(          title+"avg_img.png", avg_img.astype("uint8"))

    len_abs_loss = get_abs_loss(gen_imgs, 100)
    for i in range(len_abs_loss.shape[0]):
        if len_abs_loss[i] > 20000:
            print(i)
    len_abs_loss_std, len_abs_loss_mean = std_mean(len_abs_loss[100:])
    plot(len_abs_loss[100:],  title+"100_abs_loss.png")
    histn(len_abs_loss[100:],  title+"hist_abs_loss.png", len_abs_loss_mean, len_abs_loss_std)

if __name__ == "__main__":
    main()