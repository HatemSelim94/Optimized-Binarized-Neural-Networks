import sys
sys.path.append("/home/hatem/Projects/server/final_repo/final_repo")
import torch
import torch.nn as nn
from torch.utils.data import Subset
import argparse
import os
from architecture_eval import Network
from processing import Transformer, DataSets
from processing.datasets import CityScapes, KittiDataset
from utilities import train_kd_v2, infer_kd, set_seeds, Clipper, DataPlotter, Tracker, model_info, clean_dir, prepare_ops_metrics, jit_save, onnx_save, layers_state_setter, LR_Scheduler
from easydict import EasyDict

parser  = argparse.ArgumentParser('DARTS')
parser.add_argument('--data_name', type=str, default='cityscapes')
#parser.add_argument('--data_path', type=str, default='../../data/cityscapes/')
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
parser.add_argument('--last_layer_binary', type=int,default=1)
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
parser.add_argument('--teacher_model_path',type=str, default='model.pt')
parser.add_argument('--teacher_binary', type=int, default=1)
parser.add_argument('--teacher_affine', type=int, default=1)
parser.add_argument('--teacher_nodes_num', type=int, default=4)
parser.add_argument('--teacher_edge_num', type=int, default=2)
parser.add_argument('--teacher_ops_num', type=int, default=6)
parser.add_argument('--teacher_cells_sequence', type=str, default='r,r,u,u')
parser.add_argument('--teacher_stem_channels', type=int, default=60)
parser.add_argument('--teacher_genotype_path', type=str, default='eval/darts/experiments/')
parser.add_argument('--teacher_use_old_ver', type=int, default=1)
parser.add_argument('--teacher_search_exp_name', type=str,default='exp1')
parser.add_argument('--teacher_dropout2d', type=float, default=0.5)
parser.add_argument('--teacher_padding_mode', type=str, default='zeros')
parser.add_argument('--teacher_binarization', type=int, default=1)
parser.add_argument('--teacher_activation', type=str, default='relu')
parser.add_argument('--teacher_first_layer_activation', type=str, default='htanh')
parser.add_argument('--teacher_use_skip', type=int, default=1)
parser.add_argument('--teacher_use_kd', type=int, default=0)
parser.add_argument('--teacher_last_layer_binary', type=int, default=0)
parser.add_argument('--teacher_last_layer_kernel_size', type=int, default=3)
parser.add_argument('--teacher_channel_expansion_ratio_r', type= float, default=2)
parser.add_argument('--teacher_channel_reduction_ratio_u', type=float, default=14)
parser.add_argument('--teacher_channel_normal_ratio_n', type=float, default=0.25)
parser.add_argument('--teacher_upsample_mode', type=str, default='bilinear')
parser.add_argument('--load_experiment_name', type=str, default='exp1')
parser.add_argument('--load_model', type=int, default=0)
parser.add_argument('--upsample_mode', type=str, default='bilinear')
args = parser.parse_args()
torch.cuda.empty_cache()
teacher_args = EasyDict()
teacher_args.binary = args.teacher_binary
teacher_args.affine = args.teacher_affine
teacher_args.nodes_num = args.teacher_nodes_num
teacher_args.edge_num = args.teacher_edge_num
teacher_args.ops_num = args.teacher_ops_num
teacher_args.network_sequence = args.teacher_cells_sequence
teacher_args.stem_channels = args.teacher_stem_channels
teacher_args.genotype_path = args.teacher_genotype_path
teacher_args.use_old_ver = args.teacher_use_old_ver
teacher_args.search_exp_name = args.teacher_search_exp_name
teacher_args.teacher = args.teacher
teacher_args.jit = args.jit 
teacher_args.onnx = args.onnx
teacher_args.network_type = args.network_type
teacher_args.dropout2d_prob = args.teacher_dropout2d
teacher_args.padding_mode = args.teacher_padding_mode
teacher_args.binarization = args.teacher_binarization
teacher_args.activation = args.teacher_activation
teacher_args.first_layer_activation = args.teacher_first_layer_activation
teacher_args.use_skip = args.teacher_use_skip
teacher_args.kd = args.teacher_use_kd
teacher_args.num_of_classes = args.num_of_classes
teacher_args.image_size = args.image_size
teacher_args.last_layer_kernel_size = args.teacher_last_layer_kernel_size
teacher_args.last_layer_binary = args.teacher_last_layer_binary
teacher_args.binary_aspp = args.binary_aspp
teacher_args.channel_expansion_ratio_r = args.teacher_channel_expansion_ratio_r
teacher_args.channel_reduction_ratio_u = args.teacher_channel_reduction_ratio_u
teacher_args.channel_normal_ratio_n = args.teacher_channel_normal_ratio_n
teacher_args.upsample_mode = args.teacher_upsample_mode
teacher_args.use_kd = args.teacher_use_kd
assert(args.use_old_ver != args.use_skip)


def main():
    set_seeds(args.seed)
    clean_dir(args)
    if not torch.cuda.is_available():
        sys.exit(1)
    train_transforms = Transformer.get_transforms({'normalize':{'mean':CityScapes.mean,'std':CityScapes.std}, 'resize':{'size':[args.image_size,args.image_size]},'random_horizontal_flip':{'flip_prob':0.2}})
    val_transforms = Transformer.get_transforms({'normalize':{'mean':CityScapes.mean,'std':CityScapes.std},'resize':{'size':[args.image_size,args.image_size]}})
    plot_transforms = Transformer.get_transforms({'resize':{'size':[args.image_size,args.image_size]}})
    train_dataset = DataSets.get_dataset(args.data_name, no_of_classes=args.num_of_classes, transforms=train_transforms)
    classes_weights = train_dataset.loss_weights_3 if args.num_of_classes ==3 else train_dataset.loss_weights_8
    classes_weights = classes_weights.cuda() if args.use_weights else None
    criterion = nn.CrossEntropyLoss(ignore_index = CityScapes.ignore_index,weight=classes_weights, reduction='none')
    criterion = criterion.to(args.device)  
    net = Network(args).to(args.device)
    net._set_criterion(criterion)
    # teacher
    teacher_net = Network(teacher_args)
    teacher_net.load_state_dict(torch.load(args.teacher_model_path), strict=False) # load teacher
    teacher_net.to(args.device)

    input_shape = (1, 3, args.image_size, args.image_size)
    #train_transforms = Transformer.get_transforms({'normalize':{'mean':CityScapes.mean,'std':CityScapes.std}, 'resize':{'size':[448,448]},'center_crop':{'size':[args.image_size,args.image_size]},'random_horizontal_flip':{'flip_prob':0.2}})
    #val_transforms = Transformer.get_transforms({'normalize':{'mean':CityScapes.mean,'std':CityScapes.std},'resize':{'size':[448,448]},'center_crop':{'size':[args.image_size,args.image_size]}})
    if args.data_name == 'cityscapes':
        val_dataset = DataSets.get_dataset(args.data_name, no_of_classes=args.num_of_classes,split='val',transforms=val_transforms)
        train_idx = range(args.train_subset)
        val_idx = range(args.val_subset) 
        sub_train_dataset = Subset(train_dataset, [i for i in train_idx])
        sub_val_dataset = Subset(val_dataset, [i for i in val_idx])
    elif args.data_name == 'kitti':
        assert  args.train_subset + args.val_subset <= 200
        if args.load_model:
            net = Network(args)
            net._set_criterion(criterion) 
            net.load_state_dict(torch.load(os.path.join(args.experiment_path, args.load_experiment_name,'model.pt')), strict=False)
            net.to(args.device)
              
        train_idx = range(args.train_subset)
        val_idx = range(args.train_subset, min(200,args.train_subset+args.val_subset)) 
        val_dataset = DataSets.get_dataset(args.data_name, no_of_classes=args.num_of_classes,split='train',transforms=val_transforms)
        train_idx = range(args.train_subset)
        val_idx = range(args.val_subset) 
        sub_train_dataset = Subset(train_dataset, [i for i in train_idx])
        sub_val_dataset = Subset(val_dataset, [i for i in val_idx])
    train_loader = torch.utils.data.DataLoader(
                    sub_train_dataset, 
                    batch_size=args.batch_size, pin_memory = True)
    val_loader = torch.utils.data.DataLoader(
                    sub_val_dataset, 
                    batch_size= args.batch_size, pin_memory = True)
    #model_info(net, input_shape, save=True, dir=os.path.join(args.experiment_path, args.experiment_name), verbose=True)
    #prepare_ops_metrics(net, input_shape)
    num_of_classes = args.num_of_classes
    if args.lr_auto:
        lr  = 0.00004*args.batch_size/16
        params_list = [{'params': net.first_layer.parameters(), 'lr': lr},]
        params_list.append({'params': net.cells[:-1].parameters(), 'lr': lr})
        params_list.append({'params': net.cells[-1].parameters(), 'lr': lr*10})
        if hasattr(net, 'last_layer'):
            params_list.append({'params': net.last_layer.parameters(), 'lr': lr*10})
        if hasattr(net, 'binaspp'):
            params_list.append({'params': net.binaspp.parameters(), 'lr': lr*10})
        if hasattr(net, 'auxlayer'):
            params_list.append({'params': net.auxlayer.parameters(), 'lr': lr*10})
        optimizer = Clipper.get_clipped_optim(args.network_optim, [params_list, lr], {'weight_decay':args.network_optim_fp_weight_decay})

    else:
        fp_params = [p for p in net.parameters() if not hasattr(p,'bin')]
        bin_params = [p for p in net.parameters() if hasattr(p,'bin')]
        optim_args = [[{'params':fp_params, 'weight_decay':args.network_optim_fp_weight_decay,'lr':args.network_optim_fp_lr},{'params':bin_params, 'weight_decay':0,'lr':args.network_optim_bin_lr, 'betas':[args.network_optim_bin_betas, args.network_optim_bin_betas ]}]]
        #optim_args = [[{'params':fp_params, 'weight_decay':args.network_optim_fp_weight_decay,'lr':args.network_optim_fp_lr},{'params':bin_params, 'weight_decay':0.001,'lr':args.network_optim_bin_lr}]]
        optimizer = Clipper.get_clipped_optim(args.network_optim, optim_args)
    #optimizer=Clipper.get_clipped_optim('SGD',optim_args)
    #scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step:((step/args.epochs))**2)
    if args.poly_scheduler:
        scheduler= LR_Scheduler(args.epochs,optimizer, len(train_loader))
    else:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[i for i in range(20, args.epochs, args.decay_step)], gamma=args.decay_val)
    data = DataPlotter(os.path.join(args.experiment_path, args.experiment_name))
    tracker = Tracker(args.epochs)
    tracker.start()
    if args.step_two:
        layers_state_setter(net, input=True, weight=False) # binarized input, fp weights
    for epoch in range(args.epochs):
        # training
        train_miou, train_loss = train_kd_v2(train_loader, net, teacher_net,criterion, optimizer, num_of_classes, scheduler=scheduler, epoch=epoch, poly_scheduler=args.poly_scheduler)
        miou, loss= infer_kd(val_loader, net, criterion, num_of_classes=num_of_classes)
        if not args.poly_scheduler:
            scheduler.step()
        tracker.print(train_loss,train_miou, loss, miou, epoch=epoch)
        data.store(epoch, train_loss, loss, train_miou, miou)
        data.plot(mode='all', save=True, seaborn=args.seaborn_style)
        data.save_as_json()
        if args.step_two:
            if epoch == args.step_two:
                layers_state_setter(net, input=True, weight=True)
    tracker.end()
    args.use_kd = 0
    st_net = Network(args).to(args.device)
    st_net.load_state_dict(net.state_dict(), strict=False)
    if args.generate_onnx:
        args.onnx = 1
        input_shape = (1, 3, 376, 672)
        new_net=Network(args).to(args.device)
        new_net.load_state_dict(net.state_dict(), strict=False)
        onnx_save(new_net.eval(), input_shape,os.path.join(args.experiment_path, args.experiment_name))
    if args.generate_jit:
        args.jit= 1
        input_shape = (1, 3, 376, 672)
        new_net=Network(args).to(args.device)
        new_net.load_state_dict(net.state_dict(), strict=False)
        jit_save(new_net.eval(),input_shape,os.path.join(args.experiment_path, args.experiment_name))
    torch.save(st_net, os.path.join(args.experiment_path, args.experiment_name,'model.pt'))
    st_net.eval()
    test_loader = torch.utils.data.DataLoader(
                val_dataset, 
                batch_size=1)
    for i, (img, label, id) in enumerate(test_loader):
        
        if i >20 and i < 300:
            continue
        img = img.cuda()
        output = st_net(img)
        prediction = torch.argmax(output, 1)
        DataSets.plot_image_label(prediction.cpu().squeeze(dim=0), id, val_dataset, transforms=plot_transforms,show_titles=False,show=False ,save=True, save_dir=os.path.join(args.experiment_path, args.experiment_name,'prediction_samples'), plot_name= f'sample_output_{id.item()}')
        if i == 330:
            break

if __name__ == '__main__':
    main()
    
