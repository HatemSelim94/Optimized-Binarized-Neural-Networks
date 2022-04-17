import shutil
import random
from tkinter.messagebox import NO
import numpy as np
import torch
import json
import os
import matplotlib.pyplot as plt
from processing import SegMetrics
from utilities.edge_hooks import calculate_ops_metrics, calculate_min
from .ops_info import ops_counter
from .param_size import params_size_counter
from .memory_counter import max_mem_counter
from .latency import calculate_ops_latency, get_latency

class DummyScheduler:
    def __init__(self):
        pass
    def __call__(self, args):
        pass


def clean_dir(args):
    try:
        shutil.rmtree(os.path.join(args.experiment_path, args.experiment_name))
    except:
        pass
    try:
        os.remove(os.path.join(args.experiment_path, args.experiment_name))
    except:
        pass

def value_to_string(value, unit=None, precision=2):
    if (value *1e-9)//1 > 0:
            output =  f'{round(value*1e-9, precision)} G'
    elif (value *1e-6)//1 > 0:
            output = f'{round(value*1e-6, precision)} M'
    elif (value *1e-3)//1 >0:
            output = f'{round(value*1e-3, precision)} K'
    else:
            output = f'{round(value, precision)} '
    if unit is not None:
        output += unit
    return output

def model_info(net, input_shape, save=False, dir=None, verbose=True, kd=False):
    with torch.no_grad():
        mops, mbops, mflops = ops_counter(net, input_shape)
        max_mem = max_mem_counter(net, input_shape)
        model_size = value_to_string(params_size_counter(net, input_shape),unit='B') # same as model_size_2
        model_size_2 = value_to_string(sum([sum([p.numel()*32/8 for p in net.parameters() if not hasattr(p, 'bin')]), sum([p.numel()*1/8 for p in net.parameters() if hasattr(p, 'bin')])]), unit='B')
        latency_ms, fps = get_latency(net, input_shape, kd=kd)
    if verbose:
        print('\n Model Info:')
        print(f'     OPS: {mops} x10⁶')
        print(f'     FLOPS: {mflops} x10⁶  BOPS: {mbops} x10⁶')
        print(f'     Maximum Memory: {max_mem} MB')
        print(f'     Model Size: {model_size_2}')
        print(f'     Latency: {latency_ms} ms')
        print(f'     FPS: {fps}')
    if save:
        info = {'latency_ms': latency_ms, 'fps':fps, 'MOPS':mops, 'MBOPS':mbops, 'MFLOPS': mflops,'max_mem':max_mem, 'model_size':model_size_2}
        os.makedirs(dir, exist_ok=True)
        filename = os.path.join(dir, 'model_info')
        with open(filename+'.json', 'w') as f:
            json.dump(info, f) 

def set_seeds(seed=4):
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False # set to True to increase the computation speed
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def kd_loss_func(input, target):
    #print(input.shape, target.shape)
    cos_sim_loss_func = torch.nn.CosineSimilarity()
    mse_loss_func = torch.nn.MSELoss()
    return 1-cos_sim_loss_func(input, target).mean() + mse_loss_func(input, target)
        
def train_arch_kd(model_dataloader, arch_dataloader, arch_kd, criterion, optimizer,t_optimizer , epoch, both=True, num_of_classes=3,device='cuda', arch_start=20):
    arch_kd.model.train()
    arch_kd.teacher_model.train()
    loss_func = kd_loss_func
    if both:
        teacher_loss = 0
        student_loss = 0
        for step, (imgs, trgts, _) in enumerate(model_dataloader):    
            # step 1 in the algorithm (DARTS paper) 
            if epoch >arch_start:
                input_search, target_search, _ = next(iter(arch_dataloader))
                input_search = input_search.to(device)
                target_search = target_search.to(device, non_blocking = True)
                arch_kd.step(input_search, target_search)
            # step 2 in the algorithm (DARTS paper)
            imgs = imgs.to(device)
            trgts = trgts.to(device, non_blocking = True)
            optimizer.zero_grad()
            t_optimizer.zero_grad()
            outputs, int_outputs_list = arch_kd.model(imgs)
            t_outputs, t_int_outputs_list = arch_kd.teacher_model(imgs)
            #print(len(int_outputs_list), len(t_int_outputs_list))
            int_total_loss = 0
            for int_outputs, t_int_outputs in zip(int_outputs_list, t_int_outputs_list):
                int_total_loss += loss_func(int_outputs, t_int_outputs)
            torch.use_deterministic_algorithms(False)
            loss = criterion(outputs, trgts) # mean
            t_loss = criterion(t_outputs, trgts)
            teacher_loss += t_loss.item()
            student_loss += loss.item()
            if step% 10 == 0:
                print(f'Batch {step}: KD loss {int_total_loss.item():.2f} Teacher loss {teacher_loss/(step+1):0.3f} Student loss {student_loss/(step+1):0.3f}')
            total_loss = loss +t_loss+ int_total_loss
            total_loss.backward()
            torch.use_deterministic_algorithms(True)
            optimizer.step() # [-1, 1] (conv)
            t_optimizer.step()
            #train_loss += (loss.item()*imgs.shape[0]) # loss per image
            #predictions = torch.argmax(torch.softmax(outputs, dim=1), dim=1)
            #metric.update(trgts.cpu().numpy(), predictions.cpu().numpy())
        #mean_iou, _ = metric.get_iou()
        #train_loss /= len(model_dataloader.dataset) # mean loss (per image)


def train_arch(model_dataloader, arch_dataloader, arch, criterion, optimizer, epoch, both=True, num_of_classes=3,device='cuda', arch_start=20):
    metric = SegMetrics(num_of_classes)
    train_loss = 0
    arch.model.train()
    if both:
        for step, (imgs, trgts, _) in enumerate(model_dataloader):    
            # step 1 in the algorithm (DARTS paper) 
            if epoch >arch_start:
                input_search, target_search, _ = next(iter(arch_dataloader))
                input_search = input_search.to(device)
                target_search = target_search.to(device, non_blocking = True)
                arch.step(input_search, target_search)
            # step 2 in the algorithm (DARTS paper)
            imgs = imgs.to(device)
            trgts = trgts.to(device, non_blocking = True)
            optimizer.zero_grad()
            outputs = arch.model(imgs)
            torch.use_deterministic_algorithms(False)
            loss = criterion(outputs, trgts) # mean
            loss.backward()
            torch.use_deterministic_algorithms(True)
            optimizer.step() # [-1, 1] (conv)
            #train_loss += (loss.item()*imgs.shape[0]) # loss per image
            #predictions = torch.argmax(torch.softmax(outputs, dim=1), dim=1)
            #metric.update(trgts.cpu().numpy(), predictions.cpu().numpy())
        #mean_iou, _ = metric.get_iou()
        #train_loss /= len(model_dataloader.dataset) # mean loss (per image)
    else:
        mean_iou = None 
        train_loss = None
        if epoch < arch_start:
            for step, (imgs, trgts, _) in enumerate(model_dataloader):    
                imgs = imgs.to(device)
                trgts = trgts.to(device, non_blocking = True)
                optimizer.zero_grad()
                outputs = arch.model(imgs)
                loss = criterion(outputs, trgts) # mean
                loss.backward()
                optimizer.step() # [-1, 1] (conv)
        else:
            for step, (imgs, trgts, _) in enumerate(arch_dataloader):    
                imgs = imgs.to(device)
                trgts = trgts.to(device, non_blocking = True)
                arch.step(imgs, trgts)
                
                #train_loss += (loss.item()*imgs.shape[0]) # loss per image
                #predictions = torch.argmax(torch.softmax(outputs, dim=1), dim=1)
                #metric.update(trgts.cpu().numpy(), predictions.cpu().numpy())
                #mean_iou, _ = metric.get_iou()
                #train_loss /= len(model_dataloader.dataset) # mean loss (per image)
    #return round(mean_iou*100, 2), train_loss


def train(train_queue, model, criterion, optimizer, num_of_classes=3,device='cuda', scheduler=None, epoch=None, poly_scheduler=None):
  metric = SegMetrics(num_of_classes)
  scheduler = scheduler if poly_scheduler else DummyScheduler()
  train_loss = 0
  model.train()
  for step, (imgs, trgts, _) in enumerate(train_queue):
    imgs = imgs.to(device)
    trgts = trgts.to(device, non_blocking = True)
    if poly_scheduler:
        scheduler(step, epoch)
    optimizer.zero_grad()
    outputs = model(imgs)
    torch.use_deterministic_algorithms(False)
    loss = criterion(outputs, trgts)
    loss.backward()
    torch.use_deterministic_algorithms(True)
    optimizer.step() # [-1, 1] (conv)
    train_loss += (loss.item()*imgs.shape[0])
    with torch.no_grad():
      predictions = torch.softmax(outputs, dim=1)
      predictions = torch.argmax(predictions, dim=1)
    metric.update(trgts.cpu().numpy(), predictions.cpu().numpy())
  mean_iou, _ = metric.get_iou()
  train_loss /= len(train_queue.dataset)

  return round(mean_iou*100, 2),train_loss

def train_kd(train_queue, model, teacher_model, criterion, optimizer, teacher_optimizer, num_of_classes=3,device='cuda', scheduler=None, epoch=None, poly_scheduler=None):
  metric = SegMetrics(num_of_classes)
  scheduler = scheduler if poly_scheduler else DummyScheduler()
  train_loss = 0
  model.train()
  for step, (imgs, trgts, _) in enumerate(train_queue):
    imgs = imgs.to(device)
    trgts = trgts.to(device, non_blocking = True)
    if poly_scheduler:
        scheduler(step, epoch)
    optimizer.zero_grad()
    teacher_optimizer.zero_grad()
    student_output, student_losses = model(imgs)
    teacher_output, teacher_losses = teacher_model(imgs)
    torch.use_deterministic_algorithms(False)
    student_loss = criterion(student_output, trgts)
    teacher_loss = criterion(teacher_output, trgts)
    kd_loss = sum([torch.nn.functional.kl_div(teacher_losses[i], student_losses[i]) for i in range(len(student_losses))])
    total_loss = student_loss + teacher_loss + kd_loss 
    total_loss.backward()
    torch.use_deterministic_algorithms(True)
    optimizer.step() # [-1, 1] (conv)
    teacher_optimizer.step()
    train_loss += (student_loss.item()*imgs.shape[0])
    with torch.no_grad():
      predictions = torch.softmax(student_output, dim=1)
      predictions = torch.argmax(predictions, dim=1)
    metric.update(trgts.cpu().numpy(), predictions.cpu().numpy())
  mean_iou, _ = metric.get_iou()
  train_loss /= len(train_queue.dataset)

  return round(mean_iou*100, 2),train_loss

def train_kd_v2(train_queue, model, teacher_model, criterion, optimizer, num_of_classes=3,device='cuda', scheduler=None, epoch=None, poly_scheduler=None):
  metric = SegMetrics(num_of_classes)
  scheduler = scheduler if poly_scheduler else DummyScheduler()
  train_loss = 0
  model.train()
  teacher_model.eval()
  for step, (imgs, trgts, _) in enumerate(train_queue):
    imgs = imgs.to(device)
    trgts = trgts.to(device, non_blocking = True)
    if poly_scheduler:
        scheduler(step, epoch)
    optimizer.zero_grad()
    
    student_output, student_intermediate_outputs = model(imgs)
    _, teacher_intermediate_outputs = teacher_model(imgs)
    torch.use_deterministic_algorithms(False)
    student_loss = criterion(student_output, trgts)
    #teacher_loss = criterion(teacher_output, trgts)
    kd_loss = sum([torch.nn.functional.kl_div(teacher_intermediate_outputs[i], student_intermediate_outputs[i]) for i in range(len(student_intermediate_outputs))])
    total_loss = student_loss + kd_loss 
    total_loss.backward()
    torch.use_deterministic_algorithms(True)
    optimizer.step() # [-1, 1] (conv)
    train_loss += (student_loss.item()*imgs.shape[0])
    with torch.no_grad():
      predictions = torch.softmax(student_output, dim=1)
      predictions = torch.argmax(predictions, dim=1)
    metric.update(trgts.cpu().numpy(), predictions.cpu().numpy())
  mean_iou, _ = metric.get_iou()
  train_loss /= len(train_queue.dataset)

  return round(mean_iou*100, 2),train_loss

def infer_kd(valid_queue, model, criterion, num_of_classes=3, device='cuda'):
  metric = SegMetrics(num_of_classes)
  model.eval()
  val_loss = 0
  with torch.no_grad():
    for i, (imgs, trgts, _ )in enumerate(valid_queue):
      imgs = imgs.to(device)
      trgts = trgts.to(device)

      outputs,_ = model(imgs)
      torch.use_deterministic_algorithms(False)
      loss = criterion(outputs, trgts)
      torch.use_deterministic_algorithms(True)
      outputs = torch.softmax(outputs, dim=1)
      predicted = torch.argmax(outputs, dim=1)
      val_loss += (loss.item() * imgs.shape[0])
      metric.update(trgts.cpu().numpy(), predicted.cpu().numpy())
    miou, _ = metric.get_iou()
      #if step % args.report_freq == 0:
    val_loss /= len(valid_queue.dataset)
    return round(miou*100, 2), val_loss

def infer(valid_queue, model, criterion, num_of_classes=3, device='cuda'):
  metric = SegMetrics(num_of_classes)
  model.eval()
  val_loss = 0
  with torch.no_grad():
    for i, (imgs, trgts, _ )in enumerate(valid_queue):
      imgs = imgs.to(device)
      trgts = trgts.to(device)

      outputs = model(imgs)
      torch.use_deterministic_algorithms(False)
      loss = criterion(outputs, trgts)
      torch.use_deterministic_algorithms(True)
      outputs = torch.softmax(outputs, dim=1)
      predicted = torch.argmax(outputs, dim=1)
      val_loss += (loss.item() * imgs.shape[0])
      metric.update(trgts.cpu().numpy(), predicted.cpu().numpy())
    miou, _ = metric.get_iou()
      #if step % args.report_freq == 0:
    val_loss /= len(valid_queue.dataset)
    return round(miou*100, 2), val_loss
  
class lr_function:
  def __init__(self, epochs):
      self.epochs = epochs

  def __call__(self, epoch):
      return (1-epoch/self.epochs)** 3


class Clipper:
    @classmethod
    def get_clipped_optim(cls,optim, args, kwargs=None):
        clipped_optim = None
        if optim == 'Adamax':
           clipped_optim = ClippedAdamax(*args)
        elif optim == 'Adam':
            if kwargs is None:
                clipped_optim = ClippedAdam(*args)
            else:
                clipped_optim = ClippedAdam(*args, **kwargs)
        elif optim == 'SGD':
            clipped_optim = ClippedSGD(*args)
        return clipped_optim
            
class ClippedAdamax(torch.optim.Adamax):
    @torch.no_grad()
    def step(self, closure=None):
        loss = super().step(closure=closure)
        for group in self.param_groups:
            [p.data.clamp_(-1, 1) for p in group['params'] if (p.grad is not None) and (hasattr(p, 'bin'))]
        return loss

class ClippedAdam(torch.optim.Adam):
    @torch.no_grad()
    def step(self, closure=None):
        loss = super().step(closure=closure)
        for group in self.param_groups:
            [p.data.clamp_(-1, 1) for p in group['params'] if (p.grad is not None) and (hasattr(p, 'bin'))]
        return loss

class ClippedSGD(torch.optim.SGD):
    @torch.no_grad()
    def step(self, closure=None):
        loss = super().step(closure=closure)
        for group in self.param_groups:
            [p.data.clamp_(-1, 1) for p in group['params'] if (p.grad is not None) and (hasattr(p, 'bin'))]
        return loss

class Tracker:
    def __init__(self, epochs = None, folds = None) :
        assert epochs != 0
        self.epochs = epochs
        self.folds = folds
    def print(self, t_loss,t_iou, v_loss=None, v_iou=None, epoch=None, fold=None, mode=None):
        if self.folds:
            print(f'Fold {fold} [{100*((fold+1)/self.folds):0.0f}%]   Epoch {epoch} [{100*((epoch+1)/self.epochs):0.0f}%]')
            print(f'Training:   Loss:   {t_loss:0.2f}   IoU: {t_iou:0.2f}')
            print(f'Validation:   Loss: {v_loss:0.2f}   IoU: {v_iou:0.2f}\n')
        elif mode == 'val':
            print(f'Epoch {epoch} [{100*((epoch+1)/self.epochs):0.0f}%]')
            print(f'Validation:   Loss:   {v_loss:0.2f}   IoU: {v_iou:0.2f}\n')
        elif v_loss is not None:
            print(f'Epoch {epoch} [{100*((epoch+1)/self.epochs):0.0f}%]')
            print(f'Training:   Loss:   {t_loss:0.2f}   IoU: {t_iou:0.2f}')
            print(f'Validation:   Loss:   {v_loss:0.2f}   IoU: {v_iou:0.2f}\n')
        else:
            print(f'Epoch {epoch} [{100*((epoch+1)/self.epochs):0.0f}%]')
            print(f'Training:   Loss:   {t_loss:0.2f}   IoU: {t_iou:0.2f}')

    def start(self):
        print('-------------------------------------------------------------------')
        print('Experiment started')
    def end(self):
        print('Experiment finished')
        print('-------------------------------------------------------------------')


class DataPlotter:
    __loss_train = 'Training Loss'
    __loss_val = 'Validation Loss'
    __iou_train = 'Training Mean IoU'
    __iou_val = 'Validation Mean IoU'
    __epochs = 'Epochs'
    def __init__(self, dir) -> None:
        self.data = {}
        self.data[self.__epochs] = []
        self.data[self.__iou_train] =[]
        self.data[self.__iou_val] =[]
        self.data[self.__loss_train] =[]
        self.data[self.__loss_val] =[]
        self.file_path = dir
        self.loaded = False
        if os.path.isfile(dir+'data.json'):
            print('Warning! file already exits')
            self.load_json()
        else:
            #dir = '/'.join(json_file_path.split('/')[:-1])
            if not os.path.isdir(dir):
                os.makedirs(dir)
    def store(self, epoch,train_loss, val_loss, train_mean_iou, val_mean_iou):
        with torch.no_grad():
            if torch.is_tensor(train_loss):
                train_loss = train_loss.item()
            
            if torch.is_tensor(val_loss):
                val_loss = val_loss.item()
        self.data[self.__loss_train].append(train_loss)
        self.data[self.__loss_val].append(val_loss)
        self.data[self.__iou_train].append(train_mean_iou)
        self.data[self.__iou_val].append(val_mean_iou)
        self.data[self.__epochs].append(epoch)
    
    def save_as_json(self):
        if self.loaded:
            with open(os.path.join(self.file_path, '_data_new.json'), 'w', encoding='utf-8') as f:
                json.dump(self.data, f, ensure_ascii=False, indent=4)
        else:
            with open(os.path.join(self.file_path, '_data.json'), 'w', encoding='utf-8') as f:
                json.dump(self.data, f, ensure_ascii=False, indent=4)
    
    def load_json(self):
        self.loaded = True
        with open(os.path.join(self.file_path, '_data_new.json'),'r' ,encoding='utf-8') as f:
            self.data = json.load(f)

    def plot(self, mode='loss', verpose = False, save=False, dpi=400, seaborn=True):
        '''
        args:
            - mode: 'loss' loss plot only, 'iou' plot only or 'all(subplot)' loss and iou plots
            - verpose(boolean): show plots
            - save (boolean) : save figures
        '''
        #plt.gcf()
        #plt.gca()
        if seaborn:
            plt.style.use('seaborn')
        if mode  == 'loss':
            plt.plot(self.data[self.__epochs],self.data[self.__loss_train], label = self.__loss_train)
            plt.plot(self.data[self.__epochs], self.data[self.__loss_val], label= self.__loss_val)
            plt.legend()
            plt.xlabel(self.__epochs)
        elif mode == 'iou':
            plt.plot(self.data[self.__epochs], self.data[self.__iou_train], label= self.__iou_train)
            plt.plot(self.data[self.__epochs], self.data[self.__iou_val], label= self.__iou_val)
            plt.legend()
            plt.xlabel(self.__epochs)
        elif mode == 'val':
            fig, ax = plt.subplots(1,2)
            ax[0].plot(self.data[self.__epochs], self.data[self.__loss_val], label='Validation')
            ax[1].plot(self.data[self.__epochs], self.data[self.__iou_val], label= 'Validation')
            ax[0].set_xlabel(self.__epochs)
            ax[1].set_xlabel(self.__epochs)
            ax[0].set_ylabel('Loss')
            ax[1].set_ylabel('Accuracy (%)')
            ax[0].legend(loc=1)
            ax[1].legend(loc=1)
            plt.tight_layout()
            
        elif mode == 'all':
            fig, ax = plt.subplots(1,2)
            ax[0].plot(self.data[self.__epochs], self.data[self.__loss_train], label = self.__loss_train)
            ax[0].plot(self.data[self.__epochs], self.data[self.__loss_val], label= self.__loss_val)
            ax[1].plot(self.data[self.__epochs], self.data[self.__iou_train], label= self.__iou_train)
            ax[1].plot(self.data[self.__epochs], self.data[self.__iou_val], label= self.__iou_val)
            ax[0].legend()
            ax[1].legend()
            ax[0].set_xlabel(self.__epochs)
            ax[1].set_xlabel(self.__epochs)
            ax[0].legend(loc=1)
            ax[1].legend(loc=1)
            plt.tight_layout()
        if save:
            plt.savefig(os.path.join(self.file_path,mode)+'_plot.png', dpi =dpi)
        if verpose:
            plt.show()
        plt.close('all')


def save_model(model, epoch,scheduler, best_val_mean_iou, experiment_path, checkpoint_name='',fold=None):
    assert (best_val_mean_iou is not None) and (checkpoint_name is not None)
    DIR = os.path.join(experiment_path,'checkpoints')
    if not os.path.isdir(DIR):
        os.makedirs(DIR)
    if best_val_mean_iou:
        checkpoint_name = 'best_model'+checkpoint_name
    PATH = os.path.join(DIR, checkpoint_name+'_checkpoint.pth.tar')
    torch.save({
        'epoch': epoch,
        'fold':fold,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': scheduler.optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'val_iou': best_val_mean_iou
    }, PATH)
    return PATH

def prepare_ops_metrics(net, input_shape):
    with torch.no_grad():
        calculate_ops_metrics(net, input_shape)
        calculate_ops_latency(net, input_shape)
        calculate_min(net)

def jit_save(model, input_shape, dir,device='cuda'):
    dummy_input = torch.randn(input_shape).to(device)
    traced_obj  = torch.jit.trace(model, dummy_input)
    torch.jit.save(traced_obj, os.path.join(dir,'jit_model.pt'))

def onnx_save(model, input_shape, dir,device='cuda',opset=11):
    dummy_input = torch.randn(input_shape).to(device)
    torch.onnx.export(model, dummy_input, 
                    os.path.join(dir,'onnx_model.onnx'), input_names=['input'],output_names=['output'], export_params=True,verbose=False, opset_version=opset)


def layers_state_setter(net,input=True, weight=True):
    def recurs(net, input,weight):
        for child in net.children():
            if callable(getattr(child, 'binarize_weight',None)):
                child.binarize_weight(weight)
            if callable(getattr(child, 'binarize_input',None)):
                child.binarize_weight(input)
            recurs(child,input, weight)
    recurs(net,input, weight)
    
##################################################################################################
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Hang Zhang
## ECE Department, Rutgers University
## Email: zhang.hang@rutgers.edu
## Copyright (c) 2017
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import math

class LR_Scheduler(object):
    """Learning Rate Scheduler

    Step mode: ``lr = baselr * 0.1 ^ {floor(epoch-1 / lr_step)}``

    Cosine mode: ``lr = baselr * 0.5 * (1 + cos(iter/maxiter))``

    Poly mode: ``lr = baselr * (1 - iter/maxiter) ^ 0.9``

    Args:
        args:  :attr:`args.lr_scheduler` lr scheduler mode (`cos`, `poly`),
          :attr:`args.lr` base learning rate, :attr:`args.epochs` number of epochs,
          :attr:`args.lr_step`

        iters_per_epoch: number of iterations per epoch
    """
    def __init__(self,  num_epochs,optimizer, iters_per_epoch=0,mode='poly',
                base_lr=0.00004,lr_step=20, warmup_epochs=0):
        self.mode = mode
        print('Using {} LR Scheduler!'.format(self.mode))
        self.lr = base_lr
        if mode == 'step':
            assert lr_step
        self.lr_step = lr_step
        self.iters_per_epoch = iters_per_epoch
        self.N = num_epochs * iters_per_epoch
        self.epoch = -1
        self.warmup_iters = warmup_epochs * iters_per_epoch
        self.optimizer = optimizer
    def __call__(self, i, epoch):
        T = epoch * self.iters_per_epoch + i
        if self.mode == 'cos':
            lr = 0.5 * self.lr * (1 + math.cos(1.0 * T / self.N * math.pi))
        elif self.mode == 'poly':
            lr = self.lr * pow((1 - 1.0 * T / self.N), 0.9)
        elif self.mode == 'step':
            lr = self.lr * (0.1 ** (epoch // self.lr_step))
        else:
            raise NotImplemented
        # warm up lr schedule
        if self.warmup_iters > 0 and T < self.warmup_iters:
            lr = lr * 1.0 * T / self.warmup_iters
        if epoch > self.epoch:
            self.epoch = epoch
        assert lr >= 0
        self._adjust_learning_rate(lr)

    def _adjust_learning_rate(self, lr):
        if len(self.optimizer.param_groups) == 1:
            self.optimizer.param_groups[0]['lr'] = lr
        else:
            # enlarge the lr at the head
            self.optimizer.param_groups[0]['lr'] = lr
            for i in range(1, len(self.optimizer.param_groups)):
                self.optimizer.param_groups[i]['lr'] = lr * 10

class BnasScore:
    def __init__(self, val_loader, criterion, nclass,dummy_input_shape, device):
        self.val_loader = val_loader
        self.criterion = criterion.to(device)
        self.nclass = nclass
        self.device = device
        self.input_shape = dummy_input_shape
    def set_parameters(self, latency_gamma=0, params_delta=0, theta_ops=0, 
                        required_latency_ms=10, required_params_size_kb=100,
                        required_ops_mops = 200):
        self.latency_gamma = latency_gamma
        self.required_latency_ms = required_latency_ms
        self.params_delta = params_delta
        self.required_params_size_kb = required_params_size_kb
        self.theta_ops = theta_ops
        self.required_ops_mops = required_ops_mops

    def get_score(self, model):
        with torch.no_grad():
            miou,_ = infer(self.val_loader, model,self.criterion, self.nclass, self.device) # 0-100
            fitness = miou  
            if self.latency_gamma > 0:
                latency_ms, _ = get_latency(model, self.input_shape)
                fitness -= max(latency_ms-self.required_latency_ms, 0)*self.latency_gamma
            if self.params_delta > 0:
                model_size = params_size_counter(model, self.input_shape) # bytes
                fitness -= max((model_size/1000)-self.required_params_size_kb, 0)*self.params_delta 
            if self.theta_ops > 0:
                mops = ops_counter(model, self.input_shape)
                fitness -= max(mops-self.required_ops_mops, 0)*self.theta_ops
        return fitness

