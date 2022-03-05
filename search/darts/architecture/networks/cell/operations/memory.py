import numpy as np

def batchnorm_memory(input_shape, output_shape, affine):
    input_size = np.prod(input_shape)*32/8
    output_size = np.prod(output_shape)*32/8 
    params_size = np.prod(output_shape)*32/8 
    if affine:
        params_size *= 2
    batchnorm_mem = output_size + input_size + params_size
    return batchnorm_mem

def pool(input_shape, output_shape):
    output_size = np.prod(output_shape)*32/8 
    input_size = np.prod(input_shape)*32/8 
    avg_pool_mem = output_size + input_size + 0
    return avg_pool_mem


class MemoryCounter:
    types= {'batchnorm':batchnorm_memory, 'avgpool':pool, 'maxpool':pool}
    def __init__(self):
        pass
    @classmethod
    def count(cls, type, args):
        return cls.types[type](*args)