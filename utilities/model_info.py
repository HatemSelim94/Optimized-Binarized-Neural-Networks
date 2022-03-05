import torch
import torch.nn as nn

def model_info(model):
    fp_params_num = sum(p.numel() for p in model.parameters() if not hasattr(p, 'bin'))
    bn_params_num = sum(p.numel() for p in model.parameters() if hasattr(p, 'bin'))
    print(f'Model parameters number: {params_to_string(fp_params_num)} (fp) {params_to_string(bn_params_num)} (bn)')
    print(f'Model parameters size: {params_size_to_string(fp_params_num*32)} '+' (fp) '+f'Model parameters size: {params_size_to_string(bn_params_num)}'+' (bn)')
    print(f'Total model parameters size: {params_size_to_string(fp_params_num*32+bn_params_num)}')
    return params_size_to_string(fp_params_num*32+bn_params_num)
    
def params_size_to_string(params_num, precision=2):
    if params_num // (8*(10 ** 6)) > 0:
        return str(round(params_num / (8*(10 ** 6)), precision)) + ' MB'
    elif params_num // (8*(10 ** 3)):
        return str(round(params_num / (8*(10 ** 3)), precision)) + ' kB'
    elif params_num // 8:
        return str(round(params_num/8,precision)) + ' B'
    else:
        return str(params_num) + ' b'


def params_to_string(params_num, units=None, precision=2):
    if units is None:
        if params_num // 10 ** 6 > 0:
            return str(round(params_num / 10 ** 6, 2)) + ' M'
        elif params_num // 10 ** 3:
            return str(round(params_num / 10 ** 3, 2)) + ' k'
        else:
            return str(params_num)
    else:
        if units == 'M':
            return str(round(params_num / 10.**6, precision)) + ' ' + units
        elif units == 'K':
            return str(round(params_num / 10.**3, precision)) + ' ' + units
        else:
            return str(params_num)



def memory(model:nn.Module, input_shape):
    input = torch.randn(input_shape)
    max_mem = model.max_memory(input)
    return max_mem


def iterate_model_recurs(model:nn.Module):
    for child in model.children():
        child.do_something()
    iterate_model_recurs(child)
