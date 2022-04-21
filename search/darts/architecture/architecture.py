import torch
import torch.optim as optim

from .networks.utilities.genotype import save_genotype
import torch.nn.functional as F
class Architecture:
    def __init__(self, model, args):
        self.model = model
        self.arch_optimizer = optim.Adam(self.model.alphas, lr=args.arch_optim_lr, 
                                        betas=(args.arch_optim_beta0, args.arch_optim_beta1), eps=args.arch_optim_eps,
                                        weight_decay=args.arch_optim_weight_decay, amsgrad=args.arch_optim_amsgrad)
        self.beta = args.ops_obj_beta
        self.gamma = args.params_obj_gamma
        self.delta = args.latency_obj_delta
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=self.arch_optimizer, gamma=0.9, milestones= [i for i in range(1, args.epochs,5)])

    def step(self, arch_inputs, arch_targets):
        self.arch_optimizer.zero_grad()
        self._backward_step(arch_inputs, arch_targets)
        self.arch_optimizer.step()
        self.scheduler.step()
        #return loss * arch_inputs.shape[0]
    
    def _backward_step(self, arch_inputs, arch_targets):
        loss = self.model._loss(arch_inputs, arch_targets)
        ops_loss, params_loss, latency_loss = self.get_losses()
        #print(ops_loss, params_loss, latency_loss)
        total_loss = self.beta*ops_loss + self.gamma* params_loss + self.delta* latency_loss + loss
        total_loss.backward()
        torch.use_deterministic_algorithms(True)
        #return loss.item()
    
    def save_genotype(self, dir=None, epoch=0, nodes=4, use_old_ver=1):
        save_genotype(self.model.alphas, dir=dir, epoch=epoch, nodes=nodes, types=self.model.unique_cells,use_old_ver=use_old_ver)

    def get_losses(self, binary=True):
        ops_loss = 0
        params_loss = 0
        latency_loss = 0
        cells = self.model.cells if binary else self.model.fp_cells
        i = 0 
        for cell in cells:
            i+=1
            ops_loss += cell.forward_ops()
            params_loss += cell.forward_params()
            latency_loss += cell.forward_latency()
        #print(ops_loss, params_loss, latency_loss)
        return ops_loss/i, params_loss/i, latency_loss/i


class ArchitectureKD:
    def __init__(self, model, teacher_model, criterion, args):
        self.model = model
        self.criterion = criterion
        self.teacher_model = teacher_model
        self.arch_optimizer = optim.Adam(self.model.alphas, lr=args.arch_optim_lr, 
                                        betas=(args.arch_optim_beta0, args.arch_optim_beta1), eps=args.arch_optim_eps,
                                        weight_decay=args.arch_optim_weight_decay, amsgrad=args.arch_optim_amsgrad)
        self.teacher_arch_optimizer = optim.Adam(self.teacher_model.alphas,  lr=args.arch_optim_lr, 
                                        betas=(args.arch_optim_beta0, args.arch_optim_beta1), eps=args.arch_optim_eps,
                                        weight_decay=args.arch_optim_weight_decay, amsgrad=args.arch_optim_amsgrad)
        self.beta = args.ops_obj_beta
        self.gamma = args.params_obj_gamma
        self.delta = args.latency_obj_delta
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=self.arch_optimizer, gamma=0.9, milestones= [i for i in range(1, args.epochs,5)])
        self.teacher_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=self.arch_optimizer, gamma=0.9, milestones= [i for i in range(1, args.epochs,5)])
    
    def step(self, arch_inputs, arch_targets):
        self.arch_optimizer.zero_grad()
        self.teacher_arch_optimizer.zero_grad()
        self._backward_step(arch_inputs, arch_targets)
        self.arch_optimizer.step()
        self.teacher_arch_optimizer.step()
        self.scheduler.step()
        self.teacher_scheduler.step()
        #return loss * arch_inputs.shape[0]
    
    def _backward_step(self, arch_inputs, arch_targets):
        t_inputs = arch_inputs.clone()
        st_output, st_cell_outputs = self.model(arch_inputs)
        t_output, t_cell_outputs = self.teacher_model(t_inputs)
        torch.use_deterministic_algorithms(False) 
        st_loss = self.criterion(st_output, arch_targets)
        t_loss = self.criterion(t_output, arch_targets)
        ops_loss, params_loss, latency_loss = self.get_losses()
        st_total_loss = self.beta*ops_loss**2 + self.gamma* params_loss**2 + self.delta* latency_loss**2 + st_loss
        #total_loss = st_total_loss + t_loss + sum([kd_loss_func(st_cell_output, t_cell_output) for t_cell_output,st_cell_output in zip(t_cell_outputs,st_cell_outputs)])
        total_loss = st_total_loss + t_loss
        total_loss.backward()
        torch.use_deterministic_algorithms(True)
        #return loss.item()
    
    def save_genotype(self, dir=None, epoch=0, nodes=4, use_old_ver=1):
        save_genotype(self.model.alphas, dir=dir, epoch=epoch, nodes=nodes, types=self.model.unique_cells,use_old_ver=use_old_ver)
    
    def save_teacher_genotype(self, dir=None, epoch=0, nodes=4, use_old_ver=1):
        save_genotype(self.teacher_model.alphas, dir=dir, epoch=epoch, nodes=nodes, types=self.model.unique_cells,use_old_ver=use_old_ver)

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

def kd_loss_func(input, target):
    #print(input.shape, target.shape)
    cos_sim_loss_func = torch.nn.CosineSimilarity()
    mse_loss_func = torch.nn.MSELoss()
    #return 1-cos_sim_loss_func(input, target).mean() + mse_loss_func(input, target)
    return mse_loss_func(input, target)