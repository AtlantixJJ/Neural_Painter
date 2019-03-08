import sys
sys.path.insert(0, ".")
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import tensorflow as tf
import json
import numpy as np
import skimage.io as io
import model
import config as cfg
import loss
import trainer
from lib import files, utils, dataloader, ops

def norm01(x):
    return (x - x.min()) / (x.max() - x.min())

def gray2rgb(img):
    cmap = plt.get_cmap('jet')
    if len(img.shape) >= 4:
        ims = []
        for im in img:
            ims.append(np.delete(cmap(im[:, :, 0]), 3, 2))
        return np.stack(ims)
    elif len(img.shape) == 3 and img.shape[2] == 1:
        return np.delete(cmap(img[:, :, 0]), 3, 2)
    else:
        return np.delete(cmap(img), 3, 2)

# buzz things
rng = np.random.RandomState(3)
CONFIG = {}
with open('nim_server/config.json', 'r') as f:
    CONFIG = json.load(f)
    for x in CONFIG['models'].values():
        CONFIG = x
        break
CONFIG['model_dir'] = "success/"
TFLAGS = cfg.get_train_config(CONFIG['model_name'])
TFLAGS['batch_before'] = False
input_dim = CONFIG['input_dim']
model_dir = CONFIG['model_dir'] + CONFIG['model_name'] + CONFIG['sery']
using_cgan = CONFIG['cgan']
gen_model, gen_config, disc_model, disc_config = model.get_model(
    CONFIG['model_name'], TFLAGS)
gen_model.name = CONFIG['field_name'] + "/" + gen_model.name
disc_model.name = CONFIG['field_name'] + "/" + disc_model.name
delta =  1. - (1. / (1. + np.exp(-5.)) - 1. / (1. + np.exp(5.)))
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.InteractiveSession(config=config)

# building graph

# A basic model
z_lrs= [1e-3, 2e-3, 4e-3, 1e-2, 2e-2, 4e-2, 0.1, 0.2]
z_lr = tf.placeholder(tf.float32, [])
delta = tf.placeholder(tf.float32, [])
z_noise = tf.placeholder(tf.float32, [None, 128], name="z_noise")
target_output = tf.placeholder(tf.float32, [None, 128, 128, 3])
mask = tf.placeholder(tf.float32, [None, 128, 128, 1])
fake_sample = tf.placeholder(tf.float32, [None, 128, 128, input_dim], name="fake_sample")
if using_cgan:
    c_noise = tf.placeholder(tf.float32, [None, TFLAGS['c_len']], name="c_noise")
    c_label = tf.placeholder(tf.float32, [None, TFLAGS['c_len']], name="c_label")

    x_fake = gen_model([z_noise, c_noise])
else:
    x_fake = gen_model(z_noise)

disc_fake = disc_model(fake_sample)[0]
gen_model.set_reuse(); disc_model.set_reuse()

trans = ops.get_inverse_process_fn(kind='tanh')
gen_output = (x_fake + 1.) * 127.5 # scale to [0, 255]
# target_output = tf.clip_by_value(tf.stop_gradient(gen_output) * (1 + delta), 0, 255)
# sketch based mask
# masked_diff = (gen_output - sketch) * mask
# delta based mask
masked_diff = (gen_output - target_output) * mask
mse_batch = tf.reduce_sum(tf.abs(masked_diff), axis=[1, 2, 3]) / (tf.reduce_sum(mask, axis=[1, 2, 3]) + 1e-6)
masked_mean_diff = tf.reduce_sum(tf.abs(masked_diff)) / (tf.reduce_sum(mask) + 1e-6)
gz = tf.gradients(masked_mean_diff, [z_noise])[0]
# smooth grad
# gz = tf.sign(gz) * tf.sqrt(1e-6 + tf.abs(gz))
gz = tf.nn.l2_normalize(gz, axis=1)
#if using_cgan:
#    gc = tf.gradients(masked_mean_diff, [c_noise])[0]
#new_zs = [z_noise - lr * gz for lr in z_lrs]
#new_zs = tf.concat(new_zs, axis=0)
#c_noises = tf.concat([c_noise]*len(z_lrs), axis=0)
#new_output = (gen_model([new_zs, c_noises]) + 1.) * 127.5
new_z = z_noise - z_lr * gz
gen_model.load_from_npz(model_dir + "_gen.npz", sess)
disc_model.load_from_npz(model_dir + "_disc.npz", sess)

feed = {
    gen_model.training: False,
    disc_model.training: False,
    gen_model.keep_prob: 1.0,
    disc_model.keep_prob: 1.0
}

# generate mask
def gen_data():
    data_z = rng.normal(0, 1, size=(1, 128))
    data_c = ops.random_boolean((1, 34), if_getchu=True)

    mcx = int(rng.normal(0, 1) * 128) + 64
    mcx = min(max(mcx, 100), 28)
    mcy = int(rng.normal(0, 1) * 128) + 64
    mcy = min(max(mcy, 100), 28)
    data_mask = np.zeros((1, 128, 128, 1))
    ms = 5
    data_mask[:, mcx-ms:mcx+ms, mcy-ms:mcy+ms, :] = 1.0
    return data_z, data_c, data_mask

data_z, data_c, data_mask = gen_data()

def iterate(num=1, lr=0.4, num_iter=10):
    # get a basic result
    feed.update({
        z_noise: data_z,
        c_noise: data_c,
        mask: data_mask,
        z_lr: lr
    })

    # original sample
    data_gen = sess.run([gen_output], feed)[0]
    # target image
    data_tar = data_gen * (1 - data_mask) + np.array([0, 0, 255]) * data_mask
    data_tar[data_tar>255] = 255
    data_tar[data_tar<0] = 0
    # mean difference vector [1, 128, 128, 3]
    diff_vector = (data_tar - data_gen) * data_mask
    diff_vector = diff_vector.reshape(-1, 3).sum(0) / data_mask.sum()
    diff_vector = diff_vector.reshape(1, 1, 1, 3)

    feed.update({
        target_output: data_tar
    })

    data_new_gen = []
    for i in range(num_iter):
        cur_img, data_new_z, data_mse, data_gz = sess.run([gen_output, new_z, masked_mean_diff, gz], feed)
        data_new_gen.append(cur_img)
        feed.update({z_noise:data_new_z})
        print("=> Iter %d MSE %f Grad_Z: %f" % (i, data_mse, data_gz.max()))
    data_new_gen = np.concatenate(data_new_gen, axis=0)

    diff = np.abs(data_new_gen - data_gen).mean(3, keepdims=True)
    diff_show = gray2rgb(norm01(diff)) * 255.
    dot_similarity = (diff_vector * (data_new_gen - data_gen)).sum(3, keepdims=True)
    dot_similarity[dot_similarity < 0] = 0
    dot_similarity_show = gray2rgb(norm01(dot_similarity)) * 255.
    data_mask_show = 255 * np.concatenate([data_mask, data_mask, data_mask], axis=3)
    shows = np.concatenate([
        data_gen, data_new_gen, data_tar, diff_show, dot_similarity_show, data_mask_show])
    mosaic = ops.arbitrary_rows_cols(shows, 7, 5)
    mosaic = norm01(mosaic)
    io.imsave("mosaic_%d.jpg" % (num), mosaic)

def iterate_mov(data_z, data_c, data_mask, partition=10, num_iter=10, lr=0.2):
    threshold = 0

    # get a basic result
    feed.update({
        z_noise: data_z,
        c_noise: data_c,
        mask: data_mask,
        z_lr: lr
    })

    # original sample
    data_gen = sess.run([gen_output], feed)[0]
    # target image
    # data_tar = data_gen * (1 - data_mask) + np.array([0, 0, 255]) * data_mask
    data_tar = data_gen.copy()
    data_tar[0, :, :, :] = 0, 0, 255
    data_tar[data_tar>255] = 255
    data_tar[data_tar<0] = 0
    # mean difference vector [1, 128, 128, 3]
    diff_vector = (data_tar - data_gen) * data_mask
    diff_vector = diff_vector.reshape(-1, 3).sum(0) / data_mask.sum()
    diff_vector = diff_vector.reshape(1, 1, 1, 3)

    feed.update({
        target_output: data_tar
    })

    data_new_gen = []
    data_new_mask = []
    for i in range(num_iter):
        data_new_z = sess.run([new_z], feed)[0]
        for j in range(4):
            feed.update({z_noise: data_new_z})
            cur_img, data_new_z, data_mse, data_gz = sess.run([gen_output, new_z, masked_mean_diff, gz], feed)

        diff = np.abs(cur_img - data_gen).sum(3, keepdims=True)
        diff[data_mask > 0.5] = 0
        diff = diff.flatten()
        diff.sort()

        threshold = diff[-partition]
        diff = np.abs(cur_img - data_gen).sum(3, keepdims=True)

        data_mask[diff > threshold] = 1

        data_new_gen.append(cur_img)
        data_new_mask.append(data_mask)

        feed.update({
            z_noise: data_z,
            target_output: data_tar,
            mask: data_mask
            })
        print("=> Iter %d MSE %f Threshold %f" % (i, data_mse, threshold))
        

    data_new_gen = np.concatenate(data_new_gen, axis=0)
    data_new_mask = np.concatenate(data_new_mask, axis=0)

    diff = np.abs(data_new_gen - data_gen).mean(3, keepdims=True)
    diff_show = gray2rgb(norm01(diff)) * 255.
    dot_similarity = (diff_vector * (data_new_gen - data_gen)).sum(3, keepdims=True)
    dot_similarity_show = gray2rgb(norm01(dot_similarity)) * 255.
    data_new_mask_show = 255 * np.concatenate([data_new_mask, data_new_mask, data_new_mask], axis=3)
    shows = np.concatenate([
        data_gen, data_new_gen, data_tar, diff_show, data_new_mask_show])
    mosaic = ops.arbitrary_rows_cols(shows, 8, 5)
    mosaic = norm01(mosaic)
    io.imsave("mosaic_partition%d.jpg" % (partition), mosaic)

for i in range(10):
    data_z, data_c, data_mask = gen_data()
    iterate(i)
