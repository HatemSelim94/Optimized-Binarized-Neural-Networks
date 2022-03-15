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

def model_info(net, input_shape, save=False, dir=None, verbose=True):
    gops = ops_counter(net, input_shape)
    max_mem = max_mem_counter(net, input_shape)
    model_size = value_to_string(params_size_counter(net, input_shape),unit='B')
    latency_ms, fps = get_latency(net, input_shape)
    if verbose:
        print('\n Model Info:')
        print(f'     OPS: {gops} x10⁶')
        print(f'     Maximum Memory: {max_mem} MB')
        print(f'     Model Size: {model_size}')
        print(f'     Latency: {latency_ms} ms')
        print(f'     FPS: {fps}')
    if save:
        info = {'latency_ms': latency_ms, 'fps':fps, 'GOPS':gops, 'max_mem':max_mem, 'model_size':model_size}
        os.makedirs(dir, exist_ok=True)
        filename = os.path.join(dir, 'model_info')
        with open(filename+'.json', 'w') as f:
            json.dump(info, f) 

def set_seeds(seed=4):
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False # set to True to increase the computation speed
    #torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
            loss = criterion(outputs, trgts) # mean
            loss.backward()
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


def train(train_queue, model, criterion, optimizer, num_of_classes=3,device='cuda'):
  metric = SegMetrics(num_of_classes)
  train_loss = 0
  for step, (imgs, trgts, _) in enumerate(train_queue):
    model.train()
    imgs = imgs.to(device)
    trgts = trgts.to(device, non_blocking = True)
    #target = Variable(target, requires_grad=False).cuda(async=True)

    # get a random minibatch from the search queue with replacement
    #input_search = Variable(input_search, requires_grad=False).cuda()
    #target_search = Variable(target_search, requires_grad=False).cuda(async=True)
    
    # step 1 in the algorithm (DARTS paper) 
    # step 2 in the algorithm (DARTS paper)
    optimizer.zero_grad()
    outputs = model(imgs)
    loss = criterion(outputs, trgts)
    loss.backward()
    #nn.utils.clip_grad_value_([p for p in model.parameters() if hasattr(p, 'bin')], 1.)
    #nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
    optimizer.step() # [-1, 1] (conv)
    train_loss += (loss.item()*imgs.shape[0])
    with torch.no_grad():
      predictions = torch.softmax(outputs, dim=1)
      predictions = torch.argmax(predictions, dim=1)
    metric.update(trgts.cpu().numpy(), predictions.cpu().numpy())
  mean_iou, _ = metric.get_iou()
  train_loss /= len(train_queue.dataset)

  return round(mean_iou*100, 2),loss


def infer(valid_queue, model, criterion, num_of_classes=3, device='cuda'):
  metric = SegMetrics(num_of_classes)
  model.eval()
  val_loss = 0

  with torch.no_grad():
    for i, (imgs, trgts, _ )in enumerate(valid_queue):
      imgs = imgs.to(device)
      trgts = trgts.to(device)

      outputs = model(imgs)
      loss = criterion(outputs, trgts)
      outputs = torch.softmax(outputs, dim=1)
      predicted = torch.argmax(outputs, dim=1)
      val_loss += (loss.item() * imgs.shape[0])
      metric.update(trgts.cpu().numpy(), predicted.cpu().numpy())
    miou, _ = metric.get_iou()
      #if step % args.report_freq == 0:
    val_loss /= len(valid_queue.dataset)
    return round(miou*100, 2), loss
  
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
            clipped_optim = ClippedAdam(*args, **kwargs)
        elif optim == 'SGD':
            clipped_optim = ClippedSGD(*args, **kwargs)
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
    __iou_val = 'Validatoin Mean IoU'
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
    calculate_ops_metrics(net, input_shape)
    calculate_ops_latency(net, input_shape)
    calculate_min(net)


def get_losses(net, binary=True):
    ops_loss = 0
    params_loss = 0
    latency_loss = 0
    cells = net.cells if binary else net.fp_cells
    for cell in cells:
        ops_loss += cell.forward_ops()
        params_loss += cell.forward_params()
        latency_loss += cell.forward_latency()
    #print(ops_loss, params_loss, latency_loss)
    return ops_loss, params_loss, latency_loss