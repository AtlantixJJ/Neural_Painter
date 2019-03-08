import os
import sys
import json
import random
import numpy as np
from PIL import Image
from skimage import io
from datetime import datetime

sys.path.insert(0, '..')
CONFIG = {}
with open('config.json', 'r') as f:
    CONFIG = json.load(f)

IMG_SIZE = (CONFIG['display_size'], CONFIG['display_size'])
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = CONFIG['gpu']

from lib import optim

editors = optim.PictureOptimizerS()
for model_config in CONFIG['models'].values():
    editors.create_new_optimizer(model_config)

def model_exist(model):
    return CONFIG['models'].has_key(model)


def save_image(dirname, image, name, time):
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    time_str = time.strftime('%Y%m%d_%H%M%S_%f')
    image.save(os.path.join(dirname, '%s_%s.png' % (time_str, name)))


def get_array(model, image):
    image = image.resize(IMG_SIZE).convert('RGBA')
    w, h = image.size
    origin = np.zeros([w, h, 3])
    mask = np.zeros([w, h, 3])
    mask_image = Image.new('RGBA', (w, h))
    new_image = Image.alpha_composite(
        Image.new('RGBA', (w, h), 'white'), image)

    for i in range(w):
        for j in range(h):
            masked = image.getpixel((i, j))[3] > 0
            color = new_image.getpixel((i, j))
            origin[j, i] = color[:3]
            mask[j, i] = [masked, masked, masked]
            mask_image.putpixel((i, j), (int(masked * 255), int(masked * 255), int(masked * 255), 255))
            new_image.putpixel(
                (i, j), (color[0], color[1], color[2], int(masked * 255)))

    time = datetime.now()
    dirname = os.path.join(CONFIG['userdata_dir'], CONFIG['models'][model]['model_name'])
    save_image(dirname, mask_image, 'mask', time)
    save_image(dirname, new_image, 'sketch', time)
    return [origin, mask]


def generate_random_image(size):
    color = '#' + ''.join(random.sample('0123456789ABCDEF', 8))
    background = Image.new('RGBA', size, color)
    return background


def generate_image(model, sketch, mask, z, c):
    z = np.fromstring(z, dtype=np.float32).reshape((1, -1))
    c = np.fromstring(c, dtype=np.float32).reshape((1, -1))
    # z = np.ones(128, dtype=np.float32)
    # c = np.zeros(34, dtype=np.float32)
    # return generate_random_image(IMG_SIZE), z.tobytes(), c.tobytes()
    gen, z, c = editors.generate(CONFIG['models'][model],
        sketch, mask, z, c, file_lr_path="learning_rate.txt")
    z = np.float32(z).tobytes()
    c = np.float32(c).tobytes()

    mode = 'RGB'

    if len(gen.shape) > 3:
        gen = gen[0, :]

    if gen.shape[-1] == 1:
        gen = gen[:, :, 0]
        mode = 'L'

    gen = np.uint8(gen)
    io.imsave("gen.png", gen)
    return Image.fromarray(np.uint8(gen), mode), z, c


def regenerate_image(model):
    # z = np.ones(128, dtype=np.float32)
    # c = np.zeros(34, dtype=np.float32)
    # return generate_random_image(IMG_SIZE), z.tobytes(), c.tobytes()
    gen, z, c = editors.generate_origin(CONFIG['models'][model])
    z = np.float32(z).tobytes()
    c = np.float32(c).tobytes()

    mode = 'RGB'

    if len(gen.shape) > 3:
        gen = gen[0, :]

    if gen.shape[-1] == 1:
        gen = gen[:, :, 0]
        mode = 'L'

    gen = np.uint8(gen)
    io.imsave("gen.png", gen)
    return Image.fromarray(np.uint8(gen), mode), z, c
