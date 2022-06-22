from processing import DataSets, Transformer, KittiRoad
import numpy as np
import torch
from torch.utils.data import Subset

def cls_weight():
    no_of_classes = 8
    dataset_name = 'cityscapes'
    data_set = DataSets.get_dataset(dataset_name, no_of_classes=no_of_classes,split='train',transforms=Transformer.get_transforms({}))
    if dataset_name == 'cityscapes':
        data_set = Subset(data_set, [i for i in range(700)])
    weights = [0]*no_of_classes
    for i in range(len(data_set)):
        _,label, _  = data_set[i]
        mask1 = label>=0
        mask2 = label <=no_of_classes
        mask = torch.logical_or(mask1, mask2)
        label = label[mask]
        classes = torch.unique(label)
        for c in classes:
            weights[c] = weights[c] + torch.sum(label == c)
    ws = []
    w_class = []
    k = 1.12
    total_labels = sum(weights)
    for w in weights:
        ws.append(w/total_labels)
    for w_c in ws:
        w_class.append(1/np.log(w_c+k))
    print(w_class)
    
def cls_weight_kitti_road():
    dataset = KittiRoad()
    weights = [0]* 2
    for i in range(len(dataset)):
        _, label = dataset[i]
        mask1= label ==0
        mask2 = label ==1
        mask = torch.logical_or(mask1, mask2)
        label = label[mask]
        classes = torch.unique(label)
        for c in classes:
            weights[c] += torch.sum(label==c)
    ws = []
    w_class = []
    k = 1.12
    total_labels = sum(weights)
    for w in weights:
        ws.append(w/total_labels)
    for w_c in ws:
        w_class.append(1/np.log(w_c+k))
    print(w_class)
    
if __name__ == '__main__':
    #cls_weight()
    cls_weight_kitti_road()