#primitives, operations, Preprocess
from .search_operations import *


NR_PRIMITIVES = [
    'max_pool_3x3',
    'avg_pool_3x3',
    'conv_1x1',
    'conv_3x3',
    'dil_conv_3x3_r4',
    'skip_connect'
]

UP_PRIMITIVES = [
    'tconv_1x1',
    'tconv_3x3',
    'tconv_5x5',
    'dil_tconv_3x3_r4',
    'dil_tconv_3x3_r6',
    'dil_tconv_3x3_r8'
]

NR_OPERATIONS= {
  'max_pool_3x3' : lambda C, stride, affine, padding_mode,jit,dropout2d: MaxPool(in_channels=C, kernel_size=3, stride=stride, padding=1,affine=affine),
  'avg_pool_3x3' : lambda C, stride, affine,padding_mode,jit,dropout2d: AvgPool(in_channels=C,kernel_size=3, stride=stride, padding=1,affine=affine),
  'conv_1x1': lambda C, stride, affine, padding_mode, jit,dropout2d: BinConv1x1(C, C, stride = stride, affine=affine, padding_mode=padding_mode,jit=jit,dropout2d=dropout2d),
  'conv_3x3': lambda C, stride, affine, padding_mode, jit,dropout2d: BinConv3x3(C, C, stride = stride, affine=affine,padding_mode=padding_mode,jit=jit,dropout2d=dropout2d),
  'dil_conv_3x3_r4' : lambda C, stride, affine, padding_mode, jit,dropout2d: BinDilConv3x3(C, C, 3, stride = stride, affine=affine, padding=4, dilation=4, padding_mode=padding_mode,jit=jit,dropout2d=dropout2d),
  'skip_connect' : lambda C, stride, affine, padding_mode, jit,dropout2d: EvalIdentity() if stride == 1 else BinConv1x1(C, C, stride = stride, affine=affine, padding_mode=padding_mode, jit=jit,dropout2d=dropout2d)
}


UP_OPERATIONS= {
  'tconv_1x1': lambda C, stride, affine, padding_mode, jit,dropout2d : BinConvT1x1(C, C, stride = stride, affine=affine, padding_mode=padding_mode, jit=jit,dropout2d=dropout2d),
  'tconv_3x3': lambda C, stride, affine, padding_mode, jit,dropout2d: BinConvT3x3(C, C, stride = stride, affine=affine,padding_mode=padding_mode, jit=jit,dropout2d=dropout2d),
  'tconv_5x5': lambda C, stride, affine, padding_mode, jit,dropout2d: BinConvT5x5(C, C, stride = stride, affine=affine,padding_mode=padding_mode, jit=jit,dropout2d=dropout2d),
  'dil_tconv_3x3_r4' : lambda C, stride, affine, padding_mode, jit,dropout2d: BinConvT3x3(C, C, stride = stride, affine=affine,padding=4,dilation=4, padding_mode=padding_mode, jit=jit,dropout2d=dropout2d),
  'dil_tconv_3x3_r6' : lambda C, stride, affine, padding_mode, jit,dropout2d: BinConvT3x3(C, C, stride = stride, affine=affine,padding=6,dilation=6,padding_mode=padding_mode, jit=jit,dropout2d=dropout2d),
  'dil_tconv_3x3_r8' : lambda C, stride, affine, padding_mode, jit,dropout2d: BinConvT3x3(C, C, stride = stride, affine=affine,padding=8,dilation=8,padding_mode=padding_mode, jit=jit,dropout2d=dropout2d),
}


Fp_NR_OPERATIONS= {
  'max_pool_3x3' : lambda C, stride, affine,padding_mode,jit,dropout2d: MaxPool(in_channels=C, kernel_size=3, stride=stride, padding=1,affine=affine),
  'avg_pool_3x3' : lambda C, stride, affine, padding_mode,jit,dropout2d: AvgPool(in_channels=C,kernel_size=3, stride=stride, padding=1,affine=affine),
  'conv_1x1': lambda C, stride, affine,padding_mode,jit,dropout2d: FpConv1x1(C, C, stride = stride, affine=affine),
  'conv_3x3': lambda C, stride, affine,padding_mode,jit,dropout2d: FpConv3x3(C, C, stride = stride, affine=affine),
  'dil_conv_3x3_r4' : lambda C, stride, affine,padding_mode,jit,dropout2d: FpDilConv3x3(C, C, kernel_size=3, stride = stride, affine=affine, padding=4, dilation=4),
  'skip_connect' : lambda C, stride, affine,padding_mode,jit,dropout2d: EvalIdentity() if stride == 1 else FpConv1x1(C, C, stride = stride, affine=affine)
}


Fp_UP_OPERATIONS= {
  'tconv_1x1': lambda C, stride, affine,padding_mode,jit,dropout2d: FpTConv1x1(C, C, stride = stride, affine=affine),
  'tconv_3x3': lambda C, stride, affine,padding_mode,jit,dropout2d: FpTConv3x3(C, C, stride = stride, affine=affine),
  'tconv_5x5': lambda C, stride, affine,padding_mode,jit,dropout2d: FpTConv5x5(C, C, stride = stride, affine=affine),
  'dil_tconv_3x3_r4' : lambda C, stride, affine,padding_mode,jit,dropout2d: FpDilTConv3x3(C, C, stride = stride, affine=affine,padding=4,dilation=4),
  'dil_tconv_3x3_r6' : lambda C, stride, affine,padding_mode,jit,dropout2d: FpDilTConv3x3(C, C, stride = stride, affine=affine,padding=6,dilation=6),
  'dil_tconv_3x3_r8' : lambda C, stride, affine,padding_mode,jit,dropout2d: FpDilTConv3x3(C, C, stride = stride, affine=affine,padding=8,dilation=8),
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
    def operations(cls, cell_type, preprocess_num,preprocess_c,c, affine, padding_mode, jit,dropout2d):
        if preprocess_num ==0:
            if cell_type == 'u':
                return BinConvT1x1(in_channels=preprocess_c,stride=2, out_channels=c, kernel_size=1, affine=affine, padding_mode=padding_mode, jit=jit,dropout2d=dropout2d)
            elif cell_type == 'r':
                return BinConv1x1(in_channels=preprocess_c,stride=2, out_channels=c, kernel_size=1, affine=affine, padding_mode=padding_mode, jit=jit,dropout2d=dropout2d)
            else:
                return BinConv1x1(in_channels=preprocess_c,stride=1, out_channels=c, kernel_size=1, affine=affine, padding_mode=padding_mode, jit=jit,dropout2d=dropout2d)
        else:
            return BinConv1x1(in_channels=preprocess_c,stride=1, out_channels=c, kernel_size=1, affine=affine, padding_mode=padding_mode, jit=jit,dropout2d=dropout2d)
    
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
    def operations(cls, cell_type, preprocess_num,preprocess_c,c, affine, padding_mode, jit,dropout2d):
        if preprocess_num ==0:
            if cell_type == 'u':
                return FpTConv1x1(in_channels=preprocess_c,stride=2, out_channels=c, kernel_size=1, affine=affine, padding_mode=padding_mode)
            elif cell_type == 'r':
                return FpConv1x1(in_channels=preprocess_c,stride=2, out_channels=c, kernel_size=1, affine=affine, padding_mode=padding_mode)
            else:
                return FpConv1x1(in_channels=preprocess_c,stride=1, out_channels=c, kernel_size=1, affine=affine, padding_mode=padding_mode)
        else:
            return FpConv1x1(in_channels=preprocess_c,stride=1, out_channels=c, kernel_size=1, affine=affine, padding_mode=padding_mode)
    
    @classmethod
    def skip(cls, cell_type, args):
        if cell_type == 'r':
            return AvgPool(*args)
        elif cell_type == 'u':
            return EvalBilinear(*args)
    




 
    
   
    
    
