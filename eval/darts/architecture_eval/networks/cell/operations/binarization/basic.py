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
    @staticmethod
    def symbolic(g:torch._C.Graph, input:torch._C.Value)-> torch._C.Value:
        #return g.op("Sign", input)
        zero = g.op('Constant', value_t=torch.tensor(0, dtype=torch.int64))
        #one = g.op('Constant', value_t=torch.tensor(1, dtype=torch.float))
        #neg_one = g.op('Constant', value_t=torch.tensor(-1, dtype=torch.float))
        #condition1 = g.op('Greater', input, zero)
        #condition2 = g.op('Less', input, zero)
        #condition3 = g.op('Equal', input, zero)
        #pos = g.op('Where',g.op('Or',condition1, condition3), one, input)
        #output = g.op('Where', condition2, neg_one, pos)
        #return output
        # or 
        one = g.op('Constant', value_t=torch.tensor(1, dtype=torch.float))
        sign = g.op('Sign', input)
        output = g.op('Where', g.op('Equal', input, zero), one, sign)
        return output



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
