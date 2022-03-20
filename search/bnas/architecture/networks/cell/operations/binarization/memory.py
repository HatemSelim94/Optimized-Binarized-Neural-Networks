import numpy as np
import math
# MobileNetV2: Inverted Residuals and Linear Bottlenecks paper
# OP memory = OP output size + OP input size + OP parameters size

def conv_memory(input_shape, output_shape, weight_shape, in_bits):
    out_bits = bits_num(np.prod(weight_shape[1:])) if in_bits == 1 else in_bits # k_h * k_w* in_ch (maximum value after conv assuming ones both weight and input are ones)
    output_size = np.prod(output_shape)*out_bits/8 # in bytes
    input_size = np.prod(input_shape)*in_bits/8
    params_size = np.prod(weight_shape)* 1 /8
    conv_mem = output_size + input_size + params_size
    return conv_mem, output_size

def batchnorm_memory(input_size, output_shape, affine):
    output_size = np.prod(output_shape)*32/8 
    params_size = np.prod(output_shape)*32/8 
    if affine:
        params_size *= 2
    batchnorm_mem = output_size + input_size + params_size
    return batchnorm_mem, output_size

def binarization_memory(input_size, output_shape):
    output_size = np.prod(output_shape)*1/8
    params_size = 0
    batchnorm_mem = output_size + input_size + params_size
    return batchnorm_mem, output_size

def tconv_memory(input_shape, output_shape, weight_shape, in_bits):
    out_bits = bits_num(np.prod(weight_shape[2:])* weight_shape[0]) if in_bits == 1 else in_bits # k_h * k_w* in_ch (maximum value after conv assuming ones in both weight and input)
    output_size = np.prod(output_shape)*out_bits/8 # in bytes
    input_size = np.prod(input_shape)*in_bits/8
    params_size = np.prod(weight_shape)* 1 /8
    conv_mem = output_size + input_size + params_size
    return conv_mem, output_size

class MemoryCounter:
    types= {'conv':conv_memory, 'tconv':tconv_memory, 'batchnorm':batchnorm_memory, 'binarize':binarization_memory}
    def __init__(self):
        pass
    @classmethod
    def count(cls, type, args):
        return cls.types[type](*args)


def bits_num(decimal_num):
    abs_decimal_num = abs(decimal_num)
    if abs_decimal_num == 0 or  abs_decimal_num == 1:
        return 1
    else:
        return 2**(math.ceil((math.log2(math.ceil(math.log2(abs_decimal_num*2)))))) # range -decimal_num to +decimal_num