import torch 
import torch.nn as nn
from torch.autograd import Function



class Binarization1(Function): # Courbariaux, Hubara
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = torch.ones_like(input)
        output[input < 0] = -1
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, =ctx.saved_tensors
        grad_input = None
        #return grad_input, None  # gradients of input and quantization(none) in forward function
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.clone()
            grad_input[torch.abs(input)>1.001] = 0
        return grad_input



class Binarization2(Function): # structured/reactnet
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = torch.ones_like(input)
        output[input < 0] = -1
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, =ctx.saved_tensors
        grad_input = None
        #return grad_input, None  # gradients of input and quantization(none) in forward function
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.clone()
            mask0 = input < -1 
            mask1 = input >=-1 and input<0
            mask2 = input < 1 and input >=0
            mask3 = input >= 1
            grad_input[mask0] = 0
            grad_input[mask1] = 2 + 2 * grad_input[mask1]
            grad_input[mask2] = 2 - 2 * grad_input[mask2]
            grad_input[mask3] = 0
        return grad_input


class BinarizationEval(nn.Module):
    def __init__(self):
        super(BinarizationEval, self).__init__()
    
    def forward(self, x):
        output = torch.ones_like(input)
        output[input < 0] = -1
        return output

