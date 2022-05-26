import torch
import torch.optim as optim

from .networks.utilities.genotype import save_genotype

class Architecture:
    def __init__(self, model, args):
        self.model = model
        self.arch_optimizer = optim.Adam(self.model.alphas, lr=args.arch_optim_lr, 
                                        betas=(args.arch_optim_beta0, args.arch_optim_beta1), eps=args.arch_optim_eps,
                                        weight_decay=args.arch_optim_weight_decay, amsgrad=args.arch_optim_amsgrad)
        
    def step(self, arch_inputs, arch_targets):
        self.arch_optimizer.zero_grad()
        self._backward_step(arch_inputs, arch_targets)
        self.arch_optimizer.step()
        #return loss * arch_inputs.shape[0]
    
    def _backward_step(self, arch_predictions, arch_targets):
        loss = self.model._loss(arch_predictions, arch_targets).mean()
        torch.use_deterministic_algorithms(False)
        loss.backward()
        torch.use_deterministic_algorithms(True)
        #return loss.item()
    
    def save_genotype(self, dir=None, epoch=0, nodes=4, use_old_ver=1):
        save_genotype(self.model.alphas, dir=dir, epoch=epoch, nodes=nodes, types=self.model.unique_cells,use_old_ver=use_old_ver)
        