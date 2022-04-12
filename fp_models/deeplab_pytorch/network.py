from turtle import forward
import torchvision 
import torch 
import torch.nn as nn

class Network(nn.Module):
    def __init__(self, args):
        super(Network, self).__init__()
        self.layers = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=False, progress=False, num_classes=args.num_of_classes)
    def forward(self, x):
        result=self.layers(x)
        return result["out"]
if __name__ == '__main__':
    n = Network()
    u=torch.randn((1,3,4,4))
    print(n(u).shape)