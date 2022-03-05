import torch
import torch.optim as optim
from .networks.utilities.genotype import save_genotype

class Architecture:
    def __init__(self, model, args):
        self.model = model
        self.arch_optimizer = optim.Adam(list(self.model.alphas), lr=args.arch_optim_lr, 
                                        betas=args.arch_optim_betas, eps=args.arch_optim_eps,
                                        weight_decay=args.arch_optim_weight_decay, amsgrad=args.arch_optim_amsgrad)
    
    def step(self, arch_inputs, arch_targets):
        self.arch_optimizer.zero_grad()
        loss = self._backward_step(arch_inputs, arch_targets)
        self.arch_optimizer.step()
        #return loss * arch_inputs.shape[0]
    
    def _backward_step(self, arch_predictions, arch_targets):
        loss = self.model._loss(arch_predictions, arch_targets)
        loss.backward()
        #return loss.item()
    
    def save_genotype(self, dir=None, epoch=0):
        save_genotype(self.model.alphas, dir, epoch)

        