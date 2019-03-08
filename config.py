import model

def hg(size, n_attr):
    if size == 64:
        map_depth = 64
        map_size = 4
        n_res = 3
        n_layers = 5
    elif size == 128:
        map_depth = 32
        n_res = 2
        map_size = 8
        n_layers = 6

    gen_model = model.simple_generator.SimpleConvolutionGenerator(
        name="G",
        map_size=map_size,
        map_depth=1024,
        n_layers=4,
        norm_mtd="default")
    disc_model = model.good_generator.GoodDiscriminator(
        name="D",
        map_depth=64,
        n_layers=n_layers,
        n_res=n_res,
        n_attr=n_attr,
        norm_mtd=None)
    return gen_model, disc_model

def hg_mask(size, n_attr):
    if size == 64:
        map_size = 4
        map_depth = 64
        n_res = 3
        n_layers = 5
    elif size == 128:
        map_depth = 32
        map_size = 8
        n_res = 2
        n_layers = 6

    gen_model = model.simple_generator.MaskConvolutionGenerator(
        name="G",
        mask_num=4,
        map_size=map_size,
        map_depth=1024,
        n_layers=4,
        norm_mtd="default")
    disc_model = model.good_generator.GoodDiscriminator(
        name="D",
        map_depth=map_depth,
        n_res=n_res,
        n_layers=n_layers,
        n_attr=n_attr,
        norm_mtd=None)
    return gen_model, disc_model

def simple(size, n_attr):
    if size == 64:
        d_layers = 5
        g_layers = 4
    elif size == 128:
        d_layers = 6
        g_layers = 5

    gen_model = model.simple_generator.SimpleConvolutionGenerator(
        name="G",
        n_layers=g_layers
        )
    disc_model = model.simple_generator.SimpleConvolutionDiscriminator(
        name="D",
        n_layers=d_layers,
        n_attr=n_attr
        )
    return gen_model, disc_model

def simple_mask(size, n_attr):
    if size == 64:
        d_layers = 5
        g_layers = 4
    elif size == 128:
        d_layers = 6
        g_layers = 5
        
    gen_model = model.simple_generator.MaskConvolutionGenerator(
        name="G",
        mask_num=9,
        n_layers=g_layers
        )
    disc_model = model.simple_generator.SimpleConvolutionDiscriminator(
        name="D",
        n_layers=d_layers,
        n_attr=n_attr)
    return gen_model, disc_model