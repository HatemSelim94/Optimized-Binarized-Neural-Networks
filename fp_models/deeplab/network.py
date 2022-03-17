import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

class ImagePooling(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1):
        super(ImagePooling, self).__init__()
        self.layers = nn.Sequential(nn.AdaptiveAvgPool2d(1), # coud be done in the forward method by torch.mean(input, (-1,-2), keepdim=True)
                      ConvBnReLUPooling(in_channels, out_channels,kernel_size, stride,padding,dilation,bias=True)
        )

    def forward(self, x):
        image_size = x.shape[-2:]
        output = self.layers(x)
        output = F.interpolate(output, size=image_size, mode='bilinear', align_corners=False)
        return output

class ASPP(nn.Module):
    def __init__(self, in_channels, rates = [6,12,18], out_channels=256):
        super(ASPP, self).__init__()
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.rates = rates
        self.__build_layers()
        self.conv1_1 = ConvBnReLU(out_channels*(len(rates)+2), out_channels, 1)
    
    def forward(self, x):
        output = []
        for layer in self.layers:
            output.append(layer(x))
        output = torch.cat(output, dim=1)
        return self.conv1_1(output)

    def __build_layers(self):
        self.layers = nn.ModuleList()
        self.layers.append(ImagePooling(self.in_channels, self.out_channels))
        for rate in self.rates:
            self.layers.append(ConvBnReLU(self.in_channels, self.out_channels, kernel_size=3, padding=rate, dilation=rate))
        self.layers.append(ConvBnReLU(self.in_channels, self.out_channels, 1))


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        model = torchvision.models.resnet34()
        #print(model)
        self.layers = nn.ModuleList()
        for i, layer in enumerate(model.children()):
            if i < 7:
                self.layers.append(layer)
        self.aspp = ASPP(256)
        #self.layers.append(model.conv1)
        #self.layers.append(model.bn1)
        #self.layers.append(model.relu)
        #self.layers.append(model.maxpool)
        #self.layers.append(model.)
        
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i == 3:
                y = x
        x = self.aspp(x)
        return x, y

class ConvBnReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, dilation=1):
        super(ConvBnReLU, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size,padding=padding,dilation=dilation, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    def forward(self, x):
        return self.layers(x)

class ConvBnReLUPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, dilation=1, bias=True):
        super(ConvBnReLUPooling, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size,padding=padding,dilation=dilation, stride=stride, bias=bias),
            #nn.LayerNorm(out_channels),
            nn.ReLU()
        )
    def forward(self, x):
        return self.layers(x)

class Decoder(nn.Module):
    def __init__(self, encoder_output_channels = 256, low_level_features_channels=64): 
        """
        Args:
            in_channels: low level features channels 
            out_channels: (256+48)
        """
        super(Decoder, self).__init__()
        self.upsampler1 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        self.conv1_1 = ConvBnReLU(low_level_features_channels, 48, 1) # reduce to 32 or 48 https://arxiv.org/pdf/1802.02611.pdf sec4.1
        self.first_conv3_3 = ConvBnReLU(encoder_output_channels+48, 256, 3, padding=1)
        self.second_conv3_3 = ConvBnReLU(256, 256, 3, padding=1)  # "it is more effective to employ two 3×3 convolution with 256 filters than using simply one or three convolutions"https://arxiv.org/pdf/1802.02611.pdf sec4.1
        self.upsampler2 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)

    def forward(self, encoder_output, low_level_features):
        upsampled_features = self.upsampler1(encoder_output)
        output = self.conv1_1(low_level_features)
        output = torch.cat((output, upsampled_features), dim=1)
        output = self.first_conv3_3(output)
        output = self.second_conv3_3(output)
        output = self.upsampler2(output)
        return output

class Network(nn.Module):
    def __init__(self, args):
        super(Network, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.final_layer = nn.Conv2d(256, args.num_of_classes, 1)
    
    def forward(self, x):
        x, y = self.encoder(x)
        x = self.decoder(x, y)
        output = self.final_layer(x)
        return output

if __name__ == '__main__':
    input = torch.randn([1,3,512,512])
    net = Network(8)
    net.eval()
    out = net(input)
    #print(net)
    print(out.shape)