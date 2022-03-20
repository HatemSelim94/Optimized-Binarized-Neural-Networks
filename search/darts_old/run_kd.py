import sys
sys.path.append("/home/hatem/Projects/server/final_repo/final_repo")
import torch
import torch.nn as nn
from torch.utils.data import Subset
import argparse
import os

from architecture import Architecture, Network_kd
from processing import Transformer, DataSets
from processing.datasets import CityScapes, KittiDataset
from utilities import infer, set_seeds, Clipper, DataPlotter, Tracker, train_arch, model_info, clean_dir

parser  = argparse.ArgumentParser('DARTS')
parser.add_argument('--data_name', type=str, default='cityscapes')
parser.add_argument('--data_path', type=str, default='../../data/cityscapes/')
#parser.add_argument('--genotype_path', type=str, default='experiments/search/three_cells_stem_1/exp1/best_genotype')
parser.add_argument('--model_path')
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--image_size', type=int, default=224)
parser.add_argument('--num_of_classes', type=int, default=3)
parser.add_argument('--train_subset', type=int, default=300)
parser.add_argument('--val_subset', type=int, default=200)
parser.add_argument('--arch_optim',type=str, default='adam')
parser.add_argument('--arch_optim_lr', type=float, default=0.01)
parser.add_argument('--arch_optim_beta0', type=float, default=0.9)
parser.add_argument('--arch_optim_beta1', type=float, default=0.999)
parser.add_argument('--arch_optim_eps', type=float, default=1e-08)
parser.add_argument('--arch_optim_weight_decay', type=float, default=5e-4)
parser.add_argument('--arch_optim_amsgrad', type=bool, default=False)
parser.add_argument('--epochs', type=int, default=40)
parser.add_argument('--network_optim', type=str, default='adamax')
parser.add_argument('--network_optim_bin_lr', type=float, default=1e-3)
parser.add_argument('--network_optim_fp_lr', type=float, default=1e-5)
parser.add_argument('--network_optim_fp_weight_decay', type=float, default=5e-4)
parser.add_argument('--network_optim_bin_betas', type=float, default=0.5)
parser.add_argument('--network_sequence', type=str, default='r,r,u,u')
parser.add_argument('--stem_channels', type=int, default=64)
parser.add_argument('--nodes_num', type=int, default=4)
parser.add_argument('--edge_num', type=int, default=2)
parser.add_argument('--ops_num', type=int, default=6)
parser.add_argument('--network_final_layer', type=str,default='bin')
#parser.add_argument('--network_scheduler', type=str, default='lambda')
parser.add_argument('--experiment_path', type=str, default='search/darts/experiments/')
parser.add_argument('--experiment_name', type=str, default='exp1')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--seed', type=int, default=4)
parser.add_argument('--arch_start', type=int, default=20)
parser.add_argument('--both', type=bool, default=False)
parser.add_argument('--affine', type=bool, default=False)
parser.add_argument('--teacher_nodes_num', type=int, default=2)
args = parser.parse_args()
torch.cuda.empty_cache()




def main():
    clean_dir(args)
    if not torch.cuda.is_available():
        sys.exit(1)
    criterion = nn.CrossEntropyLoss(ignore_index = CityScapes.ignore_index, label_smoothing=0.2)
    criterion = criterion.to(args.device)  
    net = Network_kd(args).to(args.device)
    net._set_criterion(criterion)
    model_info(net,(1, 3, args.image_size, args.image_size), save=True, dir=os.path.join(args.experiment_path, args.experiment_name), verbose=True)
    arch = Architecture(net, args)
    train_transforms = Transformer.get_transforms({'normalize':{'mean':CityScapes.mean,'std':CityScapes.std}, 'resize':{'size':[args.image_size,args.image_size]},'random_horizontal_flip':{'flip_prob':0.2}})
    val_transforms = Transformer.get_transforms({'normalize':{'mean':CityScapes.mean,'std':CityScapes.std},'resize':{'size':[args.image_size,args.image_size]}})
    train_dataset = DataSets.get_dataset(args.data_name, no_of_classes=args.num_of_classes, transforms=train_transforms)
    val_dataset = DataSets.get_dataset(args.data_name, no_of_classes=args.num_of_classes,split='val',transforms=val_transforms)
    train_idx = range(args.train_subset)
    val_idx = range(args.val_subset) 
    train_dataset = Subset(train_dataset, [i for i in train_idx])
    val_dataset = Subset(val_dataset, [i for i in val_idx])
    train_loader = torch.utils.data.DataLoader(
                    train_dataset, 
                    batch_size=args.batch_size, pin_memory = True)
    val_loader = torch.utils.data.DataLoader(
                    val_dataset, 
                    batch_size= args.batch_size, pin_memory = True)
    
    set_seeds(args.seed)  
    fp_params = [p for p in net.parameters() if not hasattr(p,'bin')]
    bin_params = [p for p in net.parameters() if hasattr(p,'bin')]
    optim_args = [[{'params':fp_params, 'weight_decay':args.network_optim_fp_weight_decay,'lr':args.network_optim_fp_lr},{'params':bin_params, 'weight_decay':0,'lr':args.network_optim_bin_lr, 'betas':[args.network_optim_bin_betas, args.network_optim_bin_betas ]}]]
    optimizer = Clipper.get_clipped_optim('Adamax', optim_args)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step:((step/args.epochs))**2)
    data = DataPlotter(os.path.join(args.experiment_path, args.experiment_name))
    tracker = Tracker(args.epochs)
    tracker.start()
    for epoch in range(args.epochs):
        # training
        train_arch(train_loader, val_loader, arch, criterion, optimizer, epoch, arch_start=args.arch_start,both=args.both)
        scheduler.step()
        miou, loss= infer(val_loader, net, criterion)
        tracker.print(0,0, loss, miou, epoch=epoch, mode='val')
        data.store(epoch, 0, loss, 0, miou)
        data.plot(mode='val', save=True, seaborn=False)
        if epoch > args.arch_start-1:
            arch.save_genotype(os.path.join(args.experiment_path, args.experiment_name), epoch, nodes=args.nodes_num)
    data.save_as_json()
    tracker.end()
  
if __name__ == '__main__':
    main()
    
