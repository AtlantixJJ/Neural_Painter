import model

def hg(size, n_attr):
    if size == 64:
        map_depth = 64
        map_size = 4
        n_res = 3
        n_layer = 5
    elif size == 128:
        map_depth = 32
        n_res = 2
        map_size = 8
        n_layer = 6

    gen_model = model.simple.SimpleConvolutionGenerator(
        name="G",
        map_size=map_size,
        map_depth=1024,
        n_layer=4,
        norm_mtd="default")
    disc_model = model.deep.DeepDiscriminator(
        name="D",
        map_depth=64,
        n_layer=n_layer,
        n_res=n_res,
        n_attr=n_attr,
        norm_mtd=None)
    return gen_model, disc_model

def hg_mask(size, n_attr):
    if size == 64:
        map_size = 4
        map_depth = 64
        n_res = 3
        n_layer = 5
    elif size == 128:
        map_depth = 32
        map_size = 8
        n_res = 2
        n_layer = 6

    gen_model = model.simple.MaskConvolutionGenerator(
        name="G",
        mask_num=4,
        map_size=map_size,
        map_depth=1024,
        n_layer=4,
        norm_mtd="default")
    disc_model = model.deep.DeepDiscriminator(
        name="D",
        map_depth=map_depth,
        n_res=n_res,
        n_layer=n_layer,
        n_attr=n_attr,
        norm_mtd=None)
    return gen_model, disc_model

def simple(size, n_attr):
    gen_model = model.simple.SimpleConvolutionGenerator(
        name="G",
        out_size=size)
    disc_model = model.simple.SimpleConvolutionDiscriminator(
        name="D",
        input_size=size,
        n_attr=n_attr)
    return gen_model, disc_model

def simple_debug(size, n_attr):
    gen_model = model.simple.SimpleConvolutionGenerator(
        name="G",
        debug=True,
        out_size=size)
    disc_model = model.simple.SimpleConvolutionDiscriminator(
        name="D",
        input_size=size,
        debug=True,
        n_attr=n_attr)
    return gen_model, disc_model

def simple_mask(size, n_attr):
    d_map_depth = 64
    if size == 64:
        map_size = 4
        d_layers = 5
        g_layers = 4
    elif size == 128:
        map_size = 4
        d_layers = 6
        g_layers = 5
        
    gen_model = model.simple.MaskConvolutionGenerator(
        name="G",
        map_size=map_size,
        mask_num=9,
        n_layer=g_layers)
    disc_model = model.simple.SimpleConvolutionDiscriminator(
        name="D",
        map_depth=d_map_depth,
        n_layer=d_layers,
        n_attr=n_attr)
    return gen_model, disc_model

def deep(size, n_attr):
    gen_model = model.deep.ResidualGenerator(
        name="G",
        out_size=size)
    disc_model = model.deep.ResidualDiscriminator(
        name="D",
        n_attr=n_attr,
        input_size=size)
    return gen_model, disc_model