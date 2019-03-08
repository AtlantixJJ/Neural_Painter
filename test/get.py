import matplotlib
matplotlib.use("agg")
import sys
sys.path.insert(0, ".")
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from lib import dataloader
from lib.ptseg import *
print(Compose)
augmentations = Compose([Scale(512), RandomRotate(10), RandomHorizontallyFlip(), RandomCrop(128)])

local_path = "/home/atlantix/data/cityscapes/"
dst = dataloader.cityscapesLoader(local_path, is_transform=True, augmentations=augmentations)
bs = 4
trainloader = DataLoader(dst, batch_size=bs, num_workers=0)
for i, data in enumerate(trainloader):
    imgs, labels = data
    import pdb;pdb.set_trace()
    imgs = imgs.numpy()[:, ::-1, :, :]
    imgs = np.transpose(imgs, [0, 2, 3, 1])
    f, axarr = plt.subplots(bs, 2)
    for j in range(bs):
        axarr[j][0].imshow(imgs[j])
        axarr[j][1].imshow(dst.decode_segmap(labels.numpy()[j]))
    plt.show()
    a = raw_input()
    if a == "ex":
        break
    else:
        plt.close()