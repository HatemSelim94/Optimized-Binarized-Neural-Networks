import sys
sys.path.append("/home/hatem/Projects/server/final_repo/final_repo")
from processing import KittiDataset, CityScapes, KITTIRoadLoader, KittiRoad, Transformer
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
def test():
    kitti_dataset = KittiDataset(no_of_classes=20)
    image,target,id = kitti_dataset[0]
    print(id)
    output = torch.ones_like(target)
    emp = []
    s = set()
    for i in range(200):
        image,target,_ = kitti_dataset[i]
        unq =torch.unique(target)
        unq_list = unq.tolist()
        for val in unq_list:
            s.add(val)
    print(s)
        #output = torch.cat([output, target], dim=1)
    image,target,id = kitti_dataset[0]
    print(torch.unique(target))
    orig_ids = kitti_dataset.convert_to_orig_ids(target.cpu(), id)
    print(torch.unique(orig_ids))

def test_kitti_test():
    test_dataset = KittiDataset(no_of_classes=20, train=False)
    dir = "eval/darts/experiments/"
    image, id = test_dataset[0]
    conv = torch.nn.Conv2d(3, 19, 1, padding=0)
    output = torch.argmax(conv(image.unsqueeze(0)),dim=1)
    output = output.permute(1,2,0).squeeze()
    print(output.shape)
    print(torch.unique(output))
    test_dataset.convert_to_orig_ids(output, id,save_dir=os.path.join(dir, 'sub_test'))
    plt.imshow(image.permute(1,2,0))
    plt.show()
    
def cityscapes_20():
    city_dataset = CityScapes(no_of_classes=20, split='test')

def my_kitti_loader_test():
    gt = np.zeros()
    transforms = Transformer.get_transforms({'random_horizontal_flip':{'flip_prob':1}})
    train_dataset = KittiRoad(transforms=transforms)
    test_dataset = KittiRoad(split='test')
    tst_img,img_name, w,h = test_dataset[0]
    img, lbl = train_dataset[0]
    print(torch.unique(lbl))
    print(img_name)
    print(w, h)
    fig, ax = plt.subplots(2,2)
    ax[0,0].imshow(img.permute(1,2,0))
    ax[0,1].imshow(lbl, cmap='gray')
    ax[1,0].imshow(tst_img.permute(1,2,0))
    #ax[1,1].imshow(test_lbl, cmap='gray')
    plt.show()

def road_loader_test():
    dataset = KITTIRoadLoader()
    for i in range(len(dataset)):
        img, lbl, name = dataset[i]
        print(name)
    print(len(dataset))
    #print(torch.unique(lbl))

    #print(img.shape)
    #print(lbl.shape)
    #fig,ax = plt.subplots(2,1)
    #ax[0].imshow(img.permute(1,2,0))
    #ax[1].imshow(lbl.permute(1,0),cmap='gray')
    #plt.show()

def data_testset():
    test_dataset = CityScapes(no_of_classes=20, split='test')
    train_dataset = CityScapes(no_of_classes=20,split='val')
    _, lbl,id = train_dataset[0]
    print(train_dataset.pred_names[id])
    print(torch.unique(lbl))
    print(torch.unique(train_dataset.to_orig_ids(lbl,index=id,save=False)))
    kitti_test = KittiDataset(no_of_classes=20, train=False)
    img,id = kitti_test[0]
    print(kitti_test.pred_names[id])
    #print(torch.unique(img))
    #print(torch.unique(train_dataset.to_orig_ids(lbl,index=id,save=False)))
def test_data_final():
    kitti_test = KittiDataset(no_of_classes=20, train=False)
    city_test = CityScapes(no_of_classes=20, split='test')
    city_img, lbl,id = city_test[0]
    kitti_img, lbl, id_ki = kitti_test[0]
    dir = "eval/darts/experiments/"
    conv = torch.nn.Conv2d(3, 19, 1, padding=0)
    output_city = torch.argmax(conv(city_img.unsqueeze(0)),dim=1)
    output_kitti = torch.argmax(conv(kitti_img.unsqueeze(0)),dim=1)
    #print(output_city.shape)
    kitti_test.save(output_kitti,id=id_ki,save_dir=os.path.join(dir, 'sub_test'))
    city_test.save(output_city, id=id, save_dir=os.path.join(dir, 'sub_test'))
    kitti_val = KittiDataset(no_of_classes=20)
    city_val = CityScapes(no_of_classes=20,split='val')
    val_img_city, lbl_city_val, id_val_city = city_val[1]
    city_val.save(lbl_city_val.unsqueeze(0), id_val_city,save_dir=os.path.join(dir, 'sub_test'))
    val_img,lbl,id = kitti_val[1]
    kitti_val.save(lbl.unsqueeze(0),id,save_dir=os.path.join(dir, 'sub_test'))


if __name__ == '__main__':
    #test_kitti_test()
    #cityscapes_20()
    #road_loader_test()
    #my_kitti_loader_test()
    test_data_final()