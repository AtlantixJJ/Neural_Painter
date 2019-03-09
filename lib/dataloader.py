import tensorflow as tf
import os, time, sys
import zipfile
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from io import BytesIO
import scipy.misc as m
from torch.utils import data
from lib.ptseg import *
from lib import ops

labels = ['blonde hair', 'brown hair', 'black hair', 'blue hair', 'pink hair', 'purple hair', 'green hair', 'red hair', 'silver hair', 'white hair', 'orange hair', 'aqua hair', 'grey hair', 'long hair', 'short hair', 'twintails', 'drill hair', 'ponytail', 'blush', 'smile', 'open mouth', 'hat', 'ribbon', 'glasses', 'blue eyes', 'red eyes', 'brown eyes', 'green eyes', 'purple eyes', 'yellow eyes', 'pink eyes', 'aqua eyes', 'black eyes', 'orange eyes']
labels = np.asarray(labels)

from torch._six import string_classes, int_classes, FileNotFoundError
def default_collate(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""

    error_msg = "batch must contain tensors, numbers, dicts or lists; found {}"
    elem_type = type(batch[0])
    if isinstance(batch[0], torch.Tensor):
        out = None
        if _use_shared_memory:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = batch[0].storage()._new_shared(numel)
            out = batch[0].new(storage)
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        elem = batch[0]
        if elem_type.__name__ == 'ndarray':
            return np.stack(batch, 0)
        if elem.shape == ():  # scalars
            py_type = float if elem.dtype.name.startswith('float') else int
            return numpy_type_map[elem.dtype.name](list(map(py_type, batch)))
    elif isinstance(batch[0], int_classes):
        return torch.LongTensor(batch)
    elif isinstance(batch[0], float):
        return torch.DoubleTensor(batch)
    elif isinstance(batch[0], string_classes):
        return batch
    elif isinstance(batch[0], list) or isinstance(batch[0], tuple):
        transposed = zip(*batch)
        return [default_collate(samples) for samples in transposed]

    raise TypeError((error_msg.format(type(batch[0]))))

class CocoDetection(Dataset):
    """`MS Coco Detection <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.

    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    def __init__(self, root, annFile, transform=None, target_transform=None):
        from pycocotools.coco import COCO
        self.root = root
        self.coco = COCO(annFile)
        self.ids = list(self.coco.imgs.keys())
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)

        path = coco.loadImgs(img_id)[0]['file_name']

        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.ids)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

class cityscapesLoader(Dataset):
    colors = [[128, 64, 128],[244, 35, 232],[70, 70, 70],[102, 102, 156],[190, 153, 153],[153, 153, 153],[250, 170, 30],[220, 220, 0],[107, 142, 35],[152, 251, 152],[0, 130, 180],[220, 20, 60],[255, 0, 0],[0, 0, 142],[0, 0, 70],[0, 60, 100],[0, 80, 100],[0, 0, 230],[119, 11, 32]]

    label_colours = dict(zip(range(19), colors))

    mean_rgb = {
        "pascal": [103.939, 116.779, 123.68],
        "cityscapes": [0.0, 0.0, 0.0],
    }  # pascal mean for PSPNet and ICNet pre-trained model

    def __init__(self,root,split="train",is_transform=False,img_size=(512, 1024),augmentations=None,img_norm=True,version="cityscapes"):
        self.class_num = 19
        self.root = root
        self.split = split
        self.is_transform = is_transform
        self.augmentations = augmentations
        self.img_norm = img_norm
        self.n_classes = 19
        self.img_size = (
            img_size if isinstance(img_size, tuple) else (img_size, img_size)
        )
        self.mean = np.array(self.mean_rgb[version])
        self.files = {}

        self.images_base = os.path.join(self.root, "leftImg8bit", self.split)
        self.annotations_base = os.path.join(
            self.root, "gtFine", self.split
        )

        self.files[split] = recursive_glob(rootdir=self.images_base, suffix=".png")

        self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
        self.valid_classes = [7,8,11,12,13,17,19,20,21,22,23,24,25,26,27,28,31,32,33]
        self.class_names = ["unlabelled","road","sidewalk","building","wall","fence","pole","traffic_light","traffic_sign","vegetation","terrain","sky","person","rider","car","truck","bus","train","motorcycle","bicycle"]

        self.ignore_index = 250
        self.class_map = dict(zip(self.valid_classes, range(19)))

        if not self.files[split]:
            raise Exception("No files for split=[%s] found in %s" % (split, self.images_base))

        print("=> Found %d %s images" % (len(self.files[split]), split))
        file_size = len(self.files[split])
        self.rng = np.random.RandomState(1234)
        self.idxs = np.array(list(range(file_size)))
        self.reset()

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, index):
        index = self.idxs[index]
        img_path = self.files[self.split][index].rstrip()
        lbl_path = os.path.join(
            self.annotations_base,
            img_path.split(os.sep)[-2],
            os.path.basename(img_path)[:-15] + "gtFine_labelIds.png")

        img = m.imread(img_path)
        img = np.array(img, dtype=np.uint8)

        lbl = m.imread(lbl_path)
        lbl = self.encode_segmap(np.array(lbl, dtype=np.uint8))
        
        imgs = []
        lbls = []

        for i in range(64):
            if self.augmentations is not None:
                img_, lbl_ = self.augmentations(img, lbl)
            if self.is_transform:
                img__, lbl__ = self.transform(img_, lbl_)

            category = lbl__.reshape([-1])
            count = np.bincount(category)[:self.class_num]
            label = np.zeros((self.class_num,))
            for i,c in enumerate(count):
                if c > 0: label[i] = 1

            imgs.append(img__)
            lbls.append(label)

        imgs = np.array(imgs)
        lbls = np.array(lbls)

        return imgs, lbls

    def transform(self, img, lbl):
        """transform
        :param img:
        :param lbl:
        """
        img = m.imresize(
            img, (self.img_size[0], self.img_size[1])
        )  # uint8 with RGB mode
        # img = img[:, :, ::-1]  # RGB -> BGR
        img = img.astype(np.float64)
        img -= self.mean
        if self.img_norm: img = img.astype(float) / 127.5 - 1

        classes = np.unique(lbl)
        lbl = lbl.astype(float)
        lbl = m.imresize(lbl, (self.img_size[0], self.img_size[1]), "nearest", mode="F")
        lbl = lbl.astype(int)

        if not np.all(classes == np.unique(lbl)):
            print("WARN: resizing labels yielded fewer classes")

        if not np.all(np.unique(lbl[lbl != self.ignore_index]) < self.n_classes):
            print("after det", classes, np.unique(lbl))
            raise ValueError("Segmentation map contained invalid class values")

        return img, lbl

    def reset(self):
        self.rng.shuffle(self.idxs)

    def decode_segmap(self, temp):
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(0, self.n_classes):
            r[temp == l] = self.label_colours[l][0]
            g[temp == l] = self.label_colours[l][1]
            b[temp == l] = self.label_colours[l][2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        return rgb

    def encode_segmap(self, mask):
        # Put all void classes to zero
        for _voidc in self.void_classes:
            mask[mask == _voidc] = self.ignore_index
        for _validc in self.valid_classes:
            mask[mask == _validc] = self.class_map[_validc]
        return mask

class ADE20KLoader(Dataset):
    def __init__(self,root,split="training",is_transform=False,img_size=512,augmentations=None,img_norm=True):
        self.class_num = 150
        self.root = root
        self.split = split
        self.is_transform = is_transform
        self.augmentations = augmentations
        self.img_norm = img_norm
        self.n_classes = 150
        self.img_size = (
            img_size if isinstance(img_size, tuple) else (img_size, img_size)
        )
        self.mean = np.array([104.00699, 116.66877, 122.67892])
        self.files = collections.defaultdict(list)

        for split in ["training", "validation"]:
            file_list = recursive_glob(
                rootdir=self.root + "images/" + self.split + "/", suffix=".jpg"
            )
            self.files[split] = file_list

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        img_path = self.files[self.split][index].rstrip()
        lbl_path = img_path[:-4] + "_seg.png"

        img = m.imread(img_path)
        img = np.array(img, dtype=np.uint8)

        lbl = m.imread(lbl_path)
        lbl = np.array(lbl, dtype=np.int32)

        for i in range(64):
            if self.augmentations is not None:
                img_, lbl_ = self.augmentations(img, lbl)
            if self.is_transform:
                img__, lbl__ = self.transform(img_, lbl_)

            category = lbl__.reshape([-1])
            count = np.bincount(category)[:self.class_num]
            label = np.zeros((self.class_num,))
            for i,c in enumerate(count):
                if c > 0: label[i] = 1

            imgs.append(img__)
            lbls.append(label)

        imgs = np.array(imgs)
        lbls = np.array(lbls)

        return img, lbl

    def transform(self, img, lbl):
        img = m.imresize(
            img, (self.img_size[0], self.img_size[1])
        )  # uint8 with RGB mode
        # img = img[:, :, ::-1]  # RGB -> BGR
        img = img.astype(np.float32)
        img -= self.mean
        if self.img_norm:
            # Resize scales images from 0 to 255, thus we need
            # to divide by 255.0
            img = img.astype(float) / 128
        # NHWC -> NCHW
        # img = img.transpose(2, 0, 1)

        lbl = self.encode_segmap(lbl)
        classes = np.unique(lbl)
        lbl = lbl.astype(float)
        lbl = m.imresize(lbl, (self.img_size[0], self.img_size[1]), "nearest", mode="F")
        lbl = lbl.astype(int)
        assert np.all(classes == np.unique(lbl))

        return img, lbl

    def encode_segmap(self, mask):
        # Refer : http://groups.csail.mit.edu/vision/datasets/ADE20K/code/loadAde20K.m
        mask = mask.astype(int)
        label_mask = np.zeros((mask.shape[0], mask.shape[1]))
        label_mask = (mask[:, :, 0] / 10.0) * 256 + mask[:, :, 1]
        return np.array(label_mask, dtype=np.uint8)

    def decode_segmap(self, temp, plot=False):
        # TODO:(@meetshah1995)
        # Verify that the color mapping is 1-to-1
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(0, self.n_classes):
            r[temp == l] = 10 * (l % 10)
            g[temp == l] = l
            b[temp == l] = 0

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        if plot:
            plt.imshow(rgb)
            plt.show()
        else:
            return rgb
            

class TFDataloader():
    def __init__(self, dataset, batch_size, num_iter, sess=None):
        """
        A workround need to specify num_iter
        """
        self.dataset = dataset
        self.num_iter = num_iter
        self.sess = sess

        self.dataset = dataset.dataset.shuffle(buffer_size=256).batch(batch_size)
        self.iterator = self.dataset.make_initializable_iterator()
        self.next_element = self.iterator.get_next()
        
    def reset(self):
        self.sess.run(self.iterator.initializer)

    def __getitem__(self, idx):
        return self.sess.run(self.next_element) #self.dataset.__getitem__(idx)
    
    def __len__(self):
        return self.num_iter #self.dataset.__len__()

class FileDataset():
    def __init__(self, data_path, img_size=(64, 64), npy_dir=None):
        self.img_size = img_size

        if ".zip" in data_path:
            self.use_zip = True
            self.data_file = zipfile.ZipFile(data_path)
            self.files = self.data_file.namelist()
            self.files.sort()
            self.files = self.files[1:]
        else:
            self.use_zip = False
            self.files = sum([[file for file in files] for path, dirs, files in os.walk(data_path) if files], [])
            self.files.sort()

        # 图片文件的列表
        filelist_t = tf.constant(self.files)
        self.file_num = len(self.files)

        # label
        label = np.load(npy_dir)
        label_t = tf.constant(label)
        self.class_num = label.shape[-1]

        dataset = tf.data.Dataset.from_tensor_slices((filelist_t, label_t))
        self.dataset = dataset.map(self._parse_function)

    def read_image_from_zip(self, filename):
        """
        An eagar function for reading image from zip
        """
        f = filename.numpy().decode("utf-8")
        return np.asarray(Image.open(BytesIO(self.data_file.read(f))))

    def _parse_function(self, filename, label):
        if self.use_zip:
            x = tf.py_function(self.read_image_from_zip, [filename], tf.float32)
        else:
            x = tf.read_file(filename)
            x = tf.image.decode_image(x)
        
        x = tf.expand_dims(x, 0)
        x = tf.image.resize_bilinear(x, (self.img_size[0], self.img_size[1]))
        x = x[0]
        x = tf.cast(x, tf.float32) / 255.0
        x = tf.image.random_brightness(x, 0.05)
        x = tf.image.random_contrast(x, 0.9, 1.1)
        x = tf.image.random_flip_left_right(x)
        x = tf.clip_by_value(x * 2 - 1, -1.0, 1.0)
        return x, label

class CelebADataset(FileDataset):
    def __init__(self, data_path, img_size=(64, 64), npy_dir=None):
        super(CelebADataset, self).__init__(data_path, img_size, npy_dir)

    # 函数将filename对应的图片文件读进来
    def _parse_function(self, filename, label):
        if self.use_zip:
            x = tf.py_function(self.read_image_from_zip, [filename], tf.float32)
        else:
            x = tf.read_file(filename)
            x = tf.image.decode_image(x)
        
        x = tf.image.crop_to_bounding_box(x, 50, 25, self.img_size[0], self.img_size[1])
        x = tf.cast(x, tf.float32) / 255.0
        x = tf.image.random_brightness(x, 0.05)
        x = tf.image.random_contrast(x, 0.9, 1.1)
        x = tf.image.random_flip_left_right(x)
        x = tf.clip_by_value(x * 2 - 1, -1.0, 1.0)
        return x, label

    def read_label(self):
        """
        For convert txt label to npy label
        """
        self.label = []
        with open(self.attr_file) as f:
            self.label_len = int(f.readline())
            self.label_name = f.readline().strip().split(" ")
            self.class_num = len(self.label_name)
            for l in f.readlines():
                l = l.strip().replace("  ", " ").split(" ")
                l = [int(i) for i in l[1:]]
                self.label.append(np.array(l))
        self.label = np.array(self.label)
        self.label[self.label==-1] = 0
        np.save(self.attr_file.replace(".txt", ""), self.label)