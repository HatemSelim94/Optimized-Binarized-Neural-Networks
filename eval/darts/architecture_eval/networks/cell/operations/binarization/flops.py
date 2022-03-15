import numpy as np

def conv_flops(output_shape, weight_shape):
    flops = np.prod(weight_shape)
    flops *= np.prod(output_shape[2:])
    flops *= output_shape[0]
    return flops

def batchnorm_flops(output_shape, affine):
    flops = np.prod(output_shape)
    if affine:
        flops *= 2
    return flops

def binarization_flops(output_shape, params_num):
    # params_num (xnor)
    flops = np.prod(output_shape)
    if (params_num is not None) and params_num != 0:
        flops *= params_num
    return flops


class FlopsCounter:
    types= {'conv':conv_flops, 'tcon':conv_flops,'batchnorm':batchnorm_flops, 'binarize':binarization_flops}
    def __init__(self):
        pass
    @classmethod
    def count(cls, type, args):
        return cls.types[type](*args)

