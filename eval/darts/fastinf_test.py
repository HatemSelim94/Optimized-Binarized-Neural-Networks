import sys
sys.path.append("/home/hatem/Projects/server/final_repo/final_repo")
import torch
import torch.nn as nn
from torch.utils.data import Subset
import argparse
import os
from fastinference import Loader
from architecture_eval import Network
from processing import Transformer, DataSets
from processing.datasets import CityScapes, KittiDataset
from utilities import train, infer, set_seeds, Clipper, DataPlotter, Tracker, model_info, clean_dir, prepare_ops_metrics, jit_save, onnx_save, layers_state_setter, LR_Scheduler

parser  = argparse.ArgumentParser('DARTS')
parser.add_argument('--data_name', type=str, default='cityscapes')
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--image_size', type=int, default=224)
parser.add_argument('--num_of_classes', type=int, default=3)
parser.add_argument('--train_subset', type=int, default=300)
parser.add_argument('--val_subset', type=int, default=200)
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
#parser.add_argument('--network_scheduler', type=str, default='lambda')
parser.add_argument('--experiment_path', type=str, default='eval/darts/experiments/')
parser.add_argument('--experiment_name', type=str, default='exp1')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--seed', type=int, default=4)
parser.add_argument('--affine', type=int, default=1)
parser.add_argument('--binary', type=int, default=1)
parser.add_argument('--last_layer_binary', type=bool,default=True)
parser.add_argument('--last_layer_kernel_size', type=int, default=3)
parser.add_argument('--genotype_path', type=str, default='search/darts/experiments/')
parser.add_argument('--search_exp_name', type=str, default='exp1')
parser.add_argument('--jit', type=int, default=0)
parser.add_argument('--padding_mode', type=str, default='zeros')
parser.add_argument('--dropout2d_prob', type=float, default=0.5)
parser.add_argument('--network_type', type=str, default='cells')
parser.add_argument('--binarization', type=int, default=1)
parser.add_argument('--first_layer_activation', type=str, default='htanh')
parser.add_argument('--activation', type=str, default='relu')
parser.add_argument('--use_skip', type=int, default=1)
parser.add_argument('--onnx', type=int, default=0)
parser.add_argument('--generate_onnx', type=int, default=0)
parser.add_argument('--generate_jit', type=int, default=0)
parser.add_argument('--use_kd', type=int, default=0)
parser.add_argument('--step_two', type=int, default=30)
parser.add_argument('--seaborn_style', type=int, default=0)
parser.add_argument('--use_old_ver', type=int, default=0)
parser.add_argument('--channel_expansion_ratio_r', type= float, default=2)
parser.add_argument('--channel_reduction_ratio_u', type=float, default=14)
parser.add_argument('--channel_normal_ratio_n', type=float, default=0.25)
parser.add_argument('--poly_scheduler', type=int, default=0)
parser.add_argument('--lr_auto',type=int, default=1)
parser.add_argument('--decay_val', type= float, default=0.01)
parser.add_argument('--decay_step', type=int, default=20)
parser.add_argument('--binary_aspp', type=int, default=1)
parser.add_argument('--use_weights', type=int, default=0)
parser.add_argument('--teacher', type=int, default=0)
parser.add_argument('--load_experiment_name', type=str, default='exp1')
parser.add_argument('--load_model', type=int, default=0)
args = parser.parse_args()
torch.cuda.empty_cache()

assert(args.use_old_ver != args.use_skip)


def main():
    set_seeds(args.seed)
    #clean_dir(args)
    if not torch.cuda.is_available():
        sys.exit(1)
    
    if args.load_model:
        net = Network(args)
        net.load_state_dict(torch.load(os.path.join(args.experiment_path, args.load_experiment_name,'model.pt')), strict=False)
        net.to(args.device)
              
    torch.save(net.state_dict(), os.path.join(args.experiment_path, args.experiment_name,'model.json'))


if __name__ == '__main__':
    main()
    
