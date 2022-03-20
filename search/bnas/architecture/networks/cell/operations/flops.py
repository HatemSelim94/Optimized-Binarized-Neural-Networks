import numpy as np

def batchnorm_flops(output_shape, affine):
    flops = np.prod(output_shape)
    if affine:
        flops *= 2
    return flops

def pooling(input_shape):
    flops = np.prod(input_shape)
    return flops

class FlopsCounter:
    types= {'batchnorm':batchnorm_flops, 'avgpool':pooling, 'maxpool':pooling}
    def __init__(self):
        pass
    @classmethod
    def count(cls, type, args):
        return cls.types[type](*args)