# https://github.com/pytorch/vision/blob/f96a8a00a1e30ed89772aaeb876fab4c6734ad9c/references/segmentation/transforms.py
import numpy as np
import random
import torch
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms as T
from torchvision.transforms import functional as F


def pad_if_smaller(img, size, fill=0):
    if not isinstance(size, list):
        size = [size, size]
    min_size = min(img.shape[-2:])
    if min_size < min(*size):
        ow, oh = img.shape[-2:]
        padh = size - oh if oh < size else 0
        padw = size - ow if ow < size else 0
        img = F.pad(img, (0, 0, padw, padh), fill=fill)
    return img

def resize_if_smaller(img, size, fill=0):
    if not isinstance(size, list):
        size = [size, size]
    min_size = min(img.shape[-2:])
    if min_size < min(*size):
        ow, oh = img.shape[-2:]
        padh = size - oh if oh < size else 0
        padw = size - ow if ow < size else 0
        img = F.pad(img, (0, 0, padw, padh), fill=fill)
    return img

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomHorizontalFlip:
    def __init__(self, flip_prob = 0.5):
        self.flip_prob = flip_prob

    def __call__(self, image, target=None):
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            if target is not None:
                target = F.hflip(target)
        return image, target


class RandomCrop:
    def __init__(self, size):
        self.size = size
        if not isinstance(size, list):
            self.size = [size, size]

    def __call__(self, image, target=None):
        #image = resize_if_smaller(image, self.size)
        #if target is not None:
        #    target = resize_if_smaller(target, self.size)
        crop_params = T.RandomCrop.get_params(image, self.size)
        image = F.crop(image, *crop_params)
        if target is not None:
            target = F.crop(target, *crop_params)
        return image, target


class CenterCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target=None):
        image = F.center_crop(image, self.size)
        if target is not None:
            target = F.center_crop(target, self.size)
        return image, target

class RandomCornerCrops:
    def __init__(self, size) -> None:
        self.size = size
    
    def __call__(self, image, target=None):
        _ , _ ,bottom_left,bottom_right, center = F.five_crop(image, size=self.size)
        rand = random.random()
        if target is not None:
            _ , _ ,bottom_left_target,bottom_right_target, center_target = F.five_crop(target, size=self.size)
        if rand <0.3:
            return bottom_left, bottom_left_target
        elif rand <0.6:
            return bottom_right, bottom_right_target
        else:
            return center, center_target


class ToTensor(object):
    def __call__(self, image, target=None):
        image = F.to_tensor(image)
        if target is not None:
            target = torch.as_tensor(np.array(target), dtype=torch.int64)
            #target = F.to_tensor(target)
        return image, target


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target=None):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target

class RandomResize():
    def __init__(self, size=None, ratio=None) -> None:
        assert size is not None or ratio is not None 
        self.size = size
        self.ratio = ratio
        if not isinstance(self.ratio, list):
                self.ratio = [self.ratio, self.ratio]

    def __call__(self, image, target):
        rand = random.random()
        if self.size is None:
            _,w,h = image.shape
            self.size = [int(w * self.ratio[0]),int(h*self.ratio[1])]
        else:
            if not isinstance(self.size, list):
                self.size = [self.size, self.size]
        if rand <0.5:
            image = F.resize(image, size=self.size, interpolation=F.InterpolationMode.BILINEAR, antialias=False)
            if target is not None:
                target = F.resize(target.unsqueeze(0),size=self.size, interpolation=F.InterpolationMode.NEAREST, antialias=False)
        return image, target.squeeze(0)

class Resize:
    def __init__(self, size) -> None:
        self.size = size
    
    def __call__(self, image, target):
        image = F.resize(image, self.size, interpolation=F.InterpolationMode.BILINEAR)
        if target is not None:
            target = F.resize(target, self.size, interpolation=F.InterpolationMode.NEAREST)
        return image, target


class Transformer:
    @staticmethod
    def get_transforms(parameters, to_tensor=True):
        transforms = {'random_horizontal_flip': RandomHorizontalFlip, 
                    'normalize':Normalize, 'resize':Resize, 'random_crop':RandomCrop,
                    'random_resize':RandomResize, 'center_crop':CenterCrop}
        list_of_transforms = []
        if to_tensor:
            list_of_transforms.append(ToTensor())
        for transform, params in parameters.items():
            if transform != 'normalize':
                list_of_transforms.insert(0, transforms[transform](**params))
            else:
                list_of_transforms.append(transforms[transform](**params))
        composed_transforms = Compose(list_of_transforms)
        return composed_transforms


def calculate_kitti_mean_std():
    from datasets import KittiDataset
    import yaml
    transforms_ = Compose([CenterCrop([374, 1240]),ToTensor()])
    data = KittiDataset(mode=1,transforms=transforms_)
    dataloader_ = DataLoader(data, len(data))
    for i, (images, _) in enumerate(dataloader_):
        #print(images.shape)
        #print(i)
        mean = images.mean(dim=[0, 2, 3])
        std = images.std(dim=[0, 2, 3])
        mean = np.round(mean.numpy(), 3)
        std = np.round(std.numpy(), 3)
        print(mean, std)
        with open('kitti_mean_std.yaml','w') as f:
            yaml.dump({'mean':mean.tolist(), 'std':std.tolist()},f)

def check_transforms():
    tt = {'random_horizontal_flip': {'flip_prob':0.3}
      ,'normalize': {'mean': [0.485, 0.456, 0.406] ,'std': [0.229, 0.224, 0.225]}, 'resize':{'size':[320, 320]}}
    tr = Transformer.get_transforms(tt)
    input = torch.rand((1, 3, 224, 224))
    for t in tr:
        print(t(input, input))
    print(tr)

if __name__ == '__main__':
    calculate_kitti_mean_std()