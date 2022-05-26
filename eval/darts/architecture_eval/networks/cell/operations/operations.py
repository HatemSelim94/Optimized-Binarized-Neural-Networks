#primitives, operations, Preprocess
from .search_operations import *


""" NR_PRIMITIVES = [
    'max_pool_3x3',
    'avg_pool_3x3',
    'conv_1x1',
    'conv_3x3',
    'dil_conv_3x3_r4',
    'skip_connect',
    'dil_conv_3x3_r8',
    'grouped_conv_3x3'
]

UP_PRIMITIVES = [
    'tconv_1x1',
    'tconv_3x3',
    'tconv_5x5',
    'dil_tconv_3x3_r4',
    'dil_tconv_3x3_r6',
    'dil_tconv_3x3_r8',
    'dil_tconv_3x3_r12',
    'dil_tconv_3x3_r16'
]
 """

NR_PRIMITIVES = [
    'conv_1x1',
    'conv_3x3',
    'dil_conv_3x3_r4',
    'dil_conv_3x3_r8',
    'dil_conv_3x3_r12',
    'grouped_conv_3x3'
]
UP_PRIMITIVES = [
    'tconv_1x1',
    'tconv_3x3',
    'dil_tconv_3x3_r4',
    'dil_tconv_3x3_r6',
    'dil_tconv_3x3_r8',
    'dil_tconv_3x3_r12'
]
NR_OPERATIONS= {
  'conv_1x1': lambda C, stride, affine, padding_mode, jit,dropout2d, binarization,activation: BinConv1x1(C, C, stride = stride, affine=affine, padding_mode=padding_mode,jit=jit,dropout2d=dropout2d,binarization=binarization,activation=activation),
  'conv_3x3': lambda C, stride, affine, padding_mode, jit,dropout2d, binarization,activation: BinConv3x3(C, C, stride = stride, affine=affine,padding_mode=padding_mode,jit=jit,dropout2d=dropout2d,binarization=binarization,activation=activation),
  'dil_conv_3x3_r4' : lambda C, stride, affine, padding_mode, jit,dropout2d, binarization,activation: BinDilConv3x3(C, C, 3, stride = stride, affine=affine, padding=4, dilation=4, padding_mode=padding_mode,jit=jit,dropout2d=dropout2d,binarization=binarization,activation=activation),
  'dil_conv_3x3_r8': lambda C, stride, affine, padding_mode, jit,dropout2d, binarization,activation: BinDilConv3x3(C, C, 3, stride = stride, affine=affine, padding=8, dilation=8, padding_mode=padding_mode,jit=jit,dropout2d=dropout2d,binarization=binarization,activation=activation),
  'dil_conv_3x3_r12': lambda C, stride, affine, padding_mode, jit,dropout2d, binarization,activation: BinDilConv3x3(C, C, 3, stride = stride, affine=affine, padding=12, dilation=12, padding_mode=padding_mode,jit=jit,dropout2d=dropout2d,binarization=binarization,activation=activation),
  'grouped_conv_3x3': lambda C, stride, affine, padding_mode, jit,dropout2d, binarization,activation: GroupedConv(C, C,stride = stride, affine=affine, padding=4, dilation=4, padding_mode=padding_mode,jit=jit,dropout2d=dropout2d,binarization=binarization,activation=activation, groups=C)
}


UP_OPERATIONS= {
  'tconv_1x1': lambda C, stride, affine, padding_mode, jit,dropout2d, binarization , activation: BinConvT1x1(C, C, stride = stride, affine=affine, padding_mode=padding_mode, jit=jit,dropout2d=dropout2d,binarization=binarization,activation=activation),
  'tconv_3x3': lambda C, stride, affine, padding_mode, jit,dropout2d, binarization, activation: BinConvT3x3(C, C, stride = stride, affine=affine,padding_mode=padding_mode, jit=jit,dropout2d=dropout2d,binarization=binarization,activation=activation),
  'dil_tconv_3x3_r4' : lambda C, stride, affine, padding_mode, jit,dropout2d, binarization, activation: BinConvT3x3(C, C, stride = stride, affine=affine,padding=4,dilation=4, padding_mode=padding_mode, jit=jit,dropout2d=dropout2d,binarization=binarization,activation=activation),
  'dil_tconv_3x3_r6' : lambda C, stride, affine, padding_mode, jit,dropout2d, binarization, activation: BinConvT3x3(C, C, stride = stride, affine=affine,padding=6,dilation=6,padding_mode=padding_mode, jit=jit,dropout2d=dropout2d,binarization=binarization,activation=activation),
  'dil_tconv_3x3_r8' : lambda C, stride, affine, padding_mode, jit,dropout2d, binarization, activation: BinConvT3x3(C, C, stride = stride, affine=affine,padding=8,dilation=8,padding_mode=padding_mode, jit=jit,dropout2d=dropout2d,binarization=binarization,activation=activation),
  'dil_tconv_3x3_r12': lambda C, stride, affine, padding_mode, jit,dropout2d, binarization, activation: BinConvT3x3(C, C, stride = stride, affine=affine,padding=12,dilation=12,padding_mode=padding_mode, jit=jit,dropout2d=dropout2d,binarization=binarization,activation=activation),
}
""" NR_OPERATIONS= {
  'max_pool_3x3' : lambda C, stride, affine, padding_mode,jit,dropout2d, binarization, activation: MaxPool(in_channels=C, kernel_size=3, stride=stride, padding=1,affine=affine,activation=activation),
  'avg_pool_3x3' : lambda C, stride, affine,padding_mode,jit,dropout2d, binarization,activation: AvgPool(in_channels=C,kernel_size=3, stride=stride, padding=1,affine=affine,activation=activation),
  'conv_1x1': lambda C, stride, affine, padding_mode, jit,dropout2d, binarization,activation: BinConv1x1(C, C, stride = stride, affine=affine, padding_mode=padding_mode,jit=jit,dropout2d=dropout2d,binarization=binarization,activation=activation),
  'conv_3x3': lambda C, stride, affine, padding_mode, jit,dropout2d, binarization,activation: BinConv3x3(C, C, stride = stride, affine=affine,padding_mode=padding_mode,jit=jit,dropout2d=dropout2d,binarization=binarization,activation=activation),
  'dil_conv_3x3_r4' : lambda C, stride, affine, padding_mode, jit,dropout2d, binarization,activation: BinDilConv3x3(C, C, 3, stride = stride, affine=affine, padding=4, dilation=4, padding_mode=padding_mode,jit=jit,dropout2d=dropout2d,binarization=binarization,activation=activation),
  'skip_connect' : lambda C, stride, affine, padding_mode, jit,dropout2d, binarization,activation: EvalIdentity() if stride == 1 else BinConv1x1(C, C, stride = stride, affine=affine, padding_mode=padding_mode, jit=jit,dropout2d=dropout2d,binarization=binarization,activation=activation),
  'dil_conv_3x3_r8': lambda C, stride, affine, padding_mode, jit,dropout2d, binarization,activation: BinDilConv3x3(C, C, 3, stride = stride, affine=affine, padding=8, dilation=8, padding_mode=padding_mode,jit=jit,dropout2d=dropout2d,binarization=binarization,activation=activation),
  'grouped_conv_3x3': lambda C, stride, affine, padding_mode, jit,dropout2d, binarization,activation: GroupedConv(C, C,stride = stride, affine=affine, padding=4, dilation=4, padding_mode=padding_mode,jit=jit,dropout2d=dropout2d,binarization=binarization,activation=activation, groups=C)
}


UP_OPERATIONS= {
  'tconv_1x1': lambda C, stride, affine, padding_mode, jit,dropout2d, binarization , activation: BinConvT1x1(C, C, stride = stride, affine=affine, padding_mode=padding_mode, jit=jit,dropout2d=dropout2d,binarization=binarization,activation=activation),
  'tconv_3x3': lambda C, stride, affine, padding_mode, jit,dropout2d, binarization, activation: BinConvT3x3(C, C, stride = stride, affine=affine,padding_mode=padding_mode, jit=jit,dropout2d=dropout2d,binarization=binarization,activation=activation),
  'tconv_5x5': lambda C, stride, affine, padding_mode, jit,dropout2d, binarization, activation: BinConvT5x5(C, C, stride = stride, affine=affine,padding_mode=padding_mode, jit=jit,dropout2d=dropout2d,binarization=binarization,activation=activation),
  'dil_tconv_3x3_r4' : lambda C, stride, affine, padding_mode, jit,dropout2d, binarization, activation: BinConvT3x3(C, C, stride = stride, affine=affine,padding=4,dilation=4, padding_mode=padding_mode, jit=jit,dropout2d=dropout2d,binarization=binarization,activation=activation),
  'dil_tconv_3x3_r6' : lambda C, stride, affine, padding_mode, jit,dropout2d, binarization, activation: BinConvT3x3(C, C, stride = stride, affine=affine,padding=6,dilation=6,padding_mode=padding_mode, jit=jit,dropout2d=dropout2d,binarization=binarization,activation=activation),
  'dil_tconv_3x3_r8' : lambda C, stride, affine, padding_mode, jit,dropout2d, binarization, activation: BinConvT3x3(C, C, stride = stride, affine=affine,padding=8,dilation=8,padding_mode=padding_mode, jit=jit,dropout2d=dropout2d,binarization=binarization,activation=activation),
  'dil_tconv_3x3_r12': lambda C, stride, affine, padding_mode, jit,dropout2d, binarization, activation: BinConvT3x3(C, C, stride = stride, affine=affine,padding=12,dilation=12,padding_mode=padding_mode, jit=jit,dropout2d=dropout2d,binarization=binarization,activation=activation),
  'dil_tconv_3x3_r16': lambda C, stride, affine, padding_mode, jit,dropout2d, binarization, activation: BinConvT3x3(C, C, stride = stride, affine=affine,padding=16,dilation=16,padding_mode=padding_mode, jit=jit,dropout2d=dropout2d,binarization=binarization,activation=activation)
} """


Fp_NR_OPERATIONS= {
  'max_pool_3x3' : lambda C, stride, affine,padding_mode,jit,dropout2d, binarization, activation: MaxPool(in_channels=C, kernel_size=3, stride=stride, padding=1,affine=affine, activation=activation),
  'avg_pool_3x3' : lambda C, stride, affine, padding_mode,jit,dropout2d, binarization, activation: AvgPool(in_channels=C,kernel_size=3, stride=stride, padding=1,affine=affine, activation=activation),
  'conv_1x1': lambda C, stride, affine,padding_mode,jit,dropout2d, binarization, activation: FpConv1x1(C, C, stride = stride, affine=affine, activation=activation),
  'conv_3x3': lambda C, stride, affine,padding_mode,jit,dropout2d, binarization, activation: FpConv3x3(C, C, stride = stride, affine=affine, activation=activation),
  'dil_conv_3x3_r4' : lambda C, stride, affine,padding_mode,jit,dropout2d, binarization,activation: FpDilConv3x3(C, C, kernel_size=3, stride = stride, affine=affine, padding=4, dilation=4, activation=activation),
  'skip_connect' : lambda C, stride, affine,padding_mode,jit,dropout2d, binarization,activation: EvalIdentity() if stride == 1 else FpConv1x1(C, C, stride = stride, affine=affine, activation=activation)
}


Fp_UP_OPERATIONS= {
  'tconv_1x1': lambda C, stride, affine,padding_mode,jit,dropout2d, binarization, activation: FpTConv1x1(C, C, stride = stride, affine=affine, activation=activation),
  'tconv_3x3': lambda C, stride, affine,padding_mode,jit,dropout2d, binarization,activation: FpTConv3x3(C, C, stride = stride, affine=affine, activation=activation),
  'tconv_5x5': lambda C, stride, affine,padding_mode,jit,dropout2d, binarization,activation: FpTConv5x5(C, C, stride = stride, affine=affine, activation=activation),
  'dil_tconv_3x3_r4' : lambda C, stride, affine,padding_mode,jit,dropout2d, binarization,activation: FpDilTConv3x3(C, C, stride = stride, affine=affine,padding=4,dilation=4, activation=activation),
  'dil_tconv_3x3_r6' : lambda C, stride, affine,padding_mode,jit,dropout2d, binarization,activation: FpDilTConv3x3(C, C, stride = stride, affine=affine,padding=6,dilation=6, activation=activation),
  'dil_tconv_3x3_r8' : lambda C, stride, affine,padding_mode,jit,dropout2d, binarization,activation: FpDilTConv3x3(C, C, stride = stride, affine=affine,padding=8,dilation=8, activation=activation),
}


class OperationsConstructor:
    ops = {'n':NR_OPERATIONS, 'r':NR_OPERATIONS, 'u':UP_OPERATIONS}
    primitives = {'n':NR_PRIMITIVES, 'r':NR_PRIMITIVES, 'u':UP_PRIMITIVES}
    def __init__(self):
        pass
    @classmethod
    def get_ops(cls, cell_type):
        return cls.ops[cell_type]
    @classmethod
    def get_primitives(cls, cell_type):
        return cls.primitives[cell_type]


class FpOperationsConstructor:
    ops = {'n':Fp_NR_OPERATIONS, 'r':Fp_NR_OPERATIONS, 'u':Fp_UP_OPERATIONS}
    primitives = {'n':NR_PRIMITIVES, 'r':NR_PRIMITIVES, 'u':UP_PRIMITIVES}
    def __init__(self):
        pass
    @classmethod
    def get_ops(cls, cell_type):
        return cls.ops[cell_type]
    @classmethod
    def get_primitives(cls, cell_type):
        return cls.primitives[cell_type]


class Preprocess:
    cell_types = ['n', 'r','u']
    def __init__(self):
        pass
    @classmethod
    def operations(cls, cell_type, preprocess_num,preprocess_c,c, affine, padding_mode, jit,dropout2d,binarization=1,activation='htanh'):
        if preprocess_num ==0:
            if cell_type == 'u':
                return BinConvT1x1(in_channels=preprocess_c,stride=2, out_channels=c, kernel_size=1, affine=affine, padding_mode=padding_mode, jit=jit,dropout2d=dropout2d,binarization=binarization,activation=activation)
            elif cell_type == 'r':
                return BinConv1x1(in_channels=preprocess_c,stride=2, out_channels=c, kernel_size=1, affine=affine, padding_mode=padding_mode, jit=jit,dropout2d=dropout2d,binarization=binarization,activation=activation)
            else:
                return BinConv1x1(in_channels=preprocess_c,stride=1, out_channels=c, kernel_size=1, affine=affine, padding_mode=padding_mode, jit=jit,dropout2d=dropout2d,binarization=binarization,activation=activation)
        else:
            return BinConv1x1(in_channels=preprocess_c,stride=1, out_channels=c, kernel_size=1, affine=affine, padding_mode=padding_mode, jit=jit,dropout2d=dropout2d,binarization=binarization,activation=activation)
    
    @classmethod
    def skip(cls, cell_type, args):
        if cell_type == 'r':
            return AvgPool(*args)
        elif cell_type == 'u':
            return EvalBilinear(*args)


class FpPreprocess:
    cell_types = ['n', 'r','u']
    def __init__(self):
        pass
    @classmethod
    def operations(cls, cell_type, preprocess_num,preprocess_c,c, affine, padding_mode, jit,dropout2d,binarization,activation):
        if preprocess_num ==0:
            if cell_type == 'u':
                return FpTConv1x1(in_channels=preprocess_c,stride=2, out_channels=c, kernel_size=1, affine=affine, padding_mode=padding_mode,activation=activation)
            elif cell_type == 'r':
                return FpConv1x1(in_channels=preprocess_c,stride=2, out_channels=c, kernel_size=1, affine=affine, padding_mode=padding_mode,activation=activation)
            else:
                return FpConv1x1(in_channels=preprocess_c,stride=1, out_channels=c, kernel_size=1, affine=affine, padding_mode=padding_mode,activation=activation)
        else:
            return FpConv1x1(in_channels=preprocess_c,stride=1, out_channels=c, kernel_size=1, affine=affine, padding_mode=padding_mode,activation=activation)
    
    @classmethod
    def skip(cls, cell_type, args):
        if cell_type == 'r':
            return AvgPool(*args)
        elif cell_type == 'u':
            return EvalBilinear(*args)
    




 
    
   
    
    
