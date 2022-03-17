import torch
import torch.optim as optim

from .networks.utilities.genotype import save_genotype

class Architecture:
    def __init__(self, model, args):
        self.model = model
        self.arch_optimizer = optim.Adam(self.model.alphas, lr=args.arch_optim_lr, 
                                        betas=(args.arch_optim_beta0, args.arch_optim_beta1), eps=args.arch_optim_eps,
                                        weight_decay=args.arch_optim_weight_decay, amsgrad=args.arch_optim_amsgrad)
        self.beta = args.ops_obj_beta
        self.gamma = args.params_obj_gamma
        self.delta = args.latency_obj_delta

    def step(self, arch_inputs, arch_targets):
        self.arch_optimizer.zero_grad()
        self._backward_step(arch_inputs, arch_targets)
        self.arch_optimizer.step()
        #return loss * arch_inputs.shape[0]
    
    def _backward_step(self, arch_predictions, arch_targets):
        loss = self.model._loss(arch_predictions, arch_targets)
        ops_loss, params_loss, latency_loss = self.get_losses()
        total_loss = self.beta*ops_loss**2 + self.gamma* params_loss**2 + self.delta* latency_loss**2 + loss
        total_loss.backward()
        #return loss.item()
    
    def save_genotype(self, dir=None, epoch=0, nodes=4):
        save_genotype(self.model.alphas, dir, epoch, nodes, self.model.unique_cells)
    
    def get_losses(self, binary=True):
        ops_loss = 0
        params_loss = 0
        latency_loss = 0
        cells = self.model.cells if binary else self.model.fp_cells
        for cell in cells:
            ops_loss += cell.forward_ops()
            params_loss += cell.forward_params()
            latency_loss += cell.forward_latency()
        #print(ops_loss, params_loss, latency_loss)
        return ops_loss, params_loss, latency_loss
        