import model

def simple(size, n_attr):
    gen_model = model.simple.SimpleConvolutionGenerator(
        name="G",
        out_size=size)
    disc_model = model.simple.SimpleConvolutionDiscriminator(
        name="D",
        input_size=size,
        n_attr=n_attr)
    return gen_model, disc_model

def sample(size, n_attr):
    gen_model = model.simple.SimpleUpsampleGenerator(
        name="G",
        out_size=size)
    disc_model = model.simple.SimpleDownsampleDiscriminator(
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

def sample_debug(size, n_attr):
    gen_model = model.simple.SimpleUpsampleGenerator(
        name="G",
        debug=True,
        out_size=size)
    disc_model = model.simple.SimpleDownsampleDiscriminator(
        name="D",
        debug=True,
        input_size=size,
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
    gen_model = model.deep.DeepGenerator(
        name="G",
        map_depth=32,
        out_size=size)
    disc_model = model.deep.DeepDiscriminator(
        name="D",
        n_attr=n_attr,
        map_depth=32,
        input_size=size)
    return gen_model, disc_model

def res(size, n_attr):
    gen_model = model.deep.ResidualGenerator(
        name="G",
        out_size=size)
    disc_model = model.deep.ResidualDiscriminator(
        name="D",
        n_attr=n_attr,
        input_size=size)
    return gen_model, disc_model