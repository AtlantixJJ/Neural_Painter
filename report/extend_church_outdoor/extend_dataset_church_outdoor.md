# Extending dataset in NIM project

## By Jianjin Xu, 2/21/2018

This is a brief report of extending experiments using NIM's code. In this experiment, `Church` dataset and `Outdoor` dataset from `iGAN` repository is used.

## Dataset information

All of the datasets are placed in `/data/datasets/nim`.

1. Outdoor dataset.

150K landscape images from MIT Places, each sized 128x128, directly downloaded from iGAN's website. Below is a sample in training process.

![](./outdoor_sample.png)

2. Church dataset.

126k church images from the LSUN challenge, each sized 128x128, directly downloaded from iGAN's website. Below is a sample in training process.

![](./church_sample.png)

Both dataset are rough, they contains irrelevant objects, are inappropriately resized or cropped. In church datasets, there are even white labels in image. Spoken frankly, it is not a good choice for `GAN` application.

## Parameters studied

WGAN/DRAGAN, goodmodel or simple model.

For WGAN:

```python
dic['sample_mode'] = "single"
dic['z_len'] = 128

dic['batch_size'] = 64
dic['input_shape'] = [128, 128, 3]
dic['learning_rate'] = 1e-4

dic['random_z_kind'] = "truncate_normal"
dic['gen_iter'] = 1
dic['disc_iter'] = 4
```

For DRAGAN, only discriminator iter is different:

```python
dic['gen_iter'] = 1
dic['disc_iter'] = 1
```

For simple model, the network structure is:

```python
def simple_outdoor_gen(dic):
    dic['z_len'] = 128
    dic['map_depth'] = 1024
    dic['map_size'] = 8
    dic['n_layer'] = 4

    dic['large_ksize'] = 9

    dic['out_fn'] = 'tanh'
    dic['norm_mtd'] = "contrib"
    dic['name'] = "SCG"
    return dic

def simple_outdoor_disc(dic):
    dic['z_len'] = 128
    dic['map_depth'] = 1024
    dic['map_size'] = 8
    dic['n_layer'] = 4

    dic['out_dim'] = 3
    dic['large_ksize'] = 9

    dic['out_fn'] = 'tanh'
    dic['norm_mtd'] = None
    dic['name'] = "SCD"
    return dic
```

For goodmodel, the network is:

```python
def goodmodel_gen(dic):
    dic['map_depth'] = 64
    dic['map_size'] = 16
    dic['batch_size'] = 64
    dic['out_dim'] = 3
    dic['name'] = "GoodGenerator"
    dic['field_name'] = "GoodGenerator"

    dic['n_enlarge'] = 4

    dic['norm_mtd'] = 'contrib'
    dic['n_layer'] = 15
    return dic

def goodmodel_disc(dic):
    dic['map_depth'] = 64 # base layer
    dic['map_size'] = 16
    dic['batch_size'] = 64
    dic['out_dim'] = 3

    dic['n_enlarge'] = 5
    dic['field_name'] = "GoodDiscriminator"
    dic['name'] = "GoodDiscriminator"

    dic['norm_mtd'] = None
    dic['n_layer'] = 15
    return dic
```

Simple Training lasts for around 200K iter, while goodmodel lasts for around 400K iter. All experiments starts linear learning rate decline after 50K, and decline to 1/20 of initial in the end of training.

Comparsion include:

1. In WGAN, simple model and goodmodel, experiment in these two datasets.


## Experiments

Goodmodel actually runs faster than simple model, probably because simple model have deep convolution filters (1024->512->256...) while goodmodel's filters are mainly 64 channels.

### WGAN training: Simple V.S. Goodmodel

1. Generated samples.

Simple model with outdoor dataset (25K / 142K / 171K / 255K)

![simple wgan outdoor 25K](wgan_outdoor_simple_sample25.png)
![simple wgan outdoor 142K](wgan_outdoor_simple_sample142.png)
![simple wgan outdoor 171K](wgan_outdoor_simple_sample171.png)
![simple wgan outdoor 255K](wgan_outdoor_simple_sample255.png)

Goodmodel with outdoor dataset (25K / 171K / 302K / 401K)

![goodmodel wgan outdoor 25K](wgan_outdoor_goodmodel_sample25.png)
![goodmodel wgan outdoor 171K](wgan_outdoor_goodmodel_sample171.png)
![goodmodel wgan outdoor 302K](wgan_outdoor_goodmodel_sample302.png)
![goodmodel wgan outdoor 401K](wgan_outdoor_goodmodel_sample401.png)

Simple model learns outdoor dataset while goodmodel fail to learn, while both models are not good enough.

Simple model with church dataset (25K / 142K / 171K / 265K)

![simple wgan church 25K](wgan_church_simple_sample25.png)
![simple wgan church 142K](wgan_church_simple_sample142.png)
![simple wgan church 171K](wgan_church_simple_sample171.png)
![simple wgan church 265K](wgan_church_simple_sample265.png)

Goodmodel with church dataset (25K / 171K / 302K / 401K)

![goodmodel wgan church 25K](wgan_church_goodmodel_sample25.png)
![goodmodel wgan church 171K](wgan_church_goodmodel_sample171.png)
![goodmodel wgan church 302K](wgan_church_goodmodel_sample302.png)
![goodmodel wgan church 401K](wgan_church_goodmodel_sample401.png)

Simple model learns church dataset while goodmodel fail to learn, while both models are not good enough. However, it seems that church dataset is easier than outdoor dataset seen from the generated samples by simple model.

2. Training loss

![](wgan_outdoor.png)

Above figure is for outdoor dataset, deep blue represents goodmodel and shallow blue is simple model. Goodmodel seems to have large instability while simple model have relatively good stability. Probably goodmodel is not suitable for this task.

![](wgan_church.png)

Above figure is for church dataset, deep red represents goodmodel and shallow red is simple model. I guess the stability of WGAN is relevant with generation quality. In this experiment, both training losses are stabler than outdoor experiment. And again, goodmodel is obviously instable while simple model is typically stable.

Goodmodel training losses have one thing in common, they converge very fast, stop increasing and starts to fluctuate. If generator is strong, the loss may go up. If discriminator is strong, the loss should drop down. But both of them fluctuate severely, I guess they are playing around with each other and learning nothing.

In conclusion, goodmodel structure is not suited for this task.

Gradient penalty:

![church](wgan_gp_church.png)

![outdoor](wgan_gp_outdoor.png)


### DRAGAN training with simple model

As a result of previous experiment, I didn't study goodmodel in natrual scenes any more.

1. Generated samples.

Simple DRAGAN outdoor (25K / 115K):

![simple dragan outdoor 25K](simple_outdoor_dragan_sample25.png)
![simple dragan outdoor 115K](simple_outdoor_dragan_sample115.png)

Simple DRAGAN church (25K / 118K):

![simple dragan church 25K](simple_church_dragan_sample25.png)
![simple dragan church 118K](simple_church_dragan_sample118.png)

DRAGAN learns much better than WGAN.

2. Training Loss.

![GAN loss](dragan_outdoor_church_gan.png)
![Gradient Penalty](dragan_outdoor_church_gp.png)

Red lines are for outdoor, green lines are for church.
## Code modified

The dataset format is `fuel` generated `HDF5` dataset, I created a new class `FuelDataset` for handling this format. It is quite straightforward

1. Open `HDF5` file with utility provided by `fuel.`.

2. Use its `get_data` method to return a particular sample.

Then I add a bunch of settings to `config`, that's all.