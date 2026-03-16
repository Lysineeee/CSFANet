import torch
import torch.nn as nn
from einops import rearrange
from torchvision.models import resnet18


class Merge_everything_model(nn.Module):
    def __init__(self, target, channels=[64, 128, 256, 512]):
        super(Merge_everything_model, self).__init__()

        self.target = target
        self.channels = channels

    def down(self, x, ratio):
        return rearrange(x, 'b c (num h) (num w) -> b (c nh nw) h w', num=ratio)

    def forward(self, T1, T2, layer):
        ratios = [2**(layer + 1), ]
        T1[0] = self.down(T1[0], ratios[0])


class ResNet18(nn.Module):
    def __init__(self, pretrained=False):
        super(ResNet18, self).__init__()
        basic_resnet = resnet18(weights='IMAGENET1K_V1' if pretrained else None)
        # print(basic_resnet)

        self.conv1 = basic_resnet.conv1
        self.bn1 = basic_resnet.bn1
        self.relu = basic_resnet.relu
        self.maxpool = basic_resnet.maxpool

        self.layer1 = basic_resnet.layer1
        self.layer2 = basic_resnet.layer2
        self.layer3 = basic_resnet.layer3
        self.layer4 = basic_resnet.layer4

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        layer1_out = self.layer1(x)
        layer2_out = self.layer2(layer1_out)
        layer3_out = self.layer3(layer2_out)
        layer4_out = self.layer4(layer3_out)

        return [layer1_out, layer2_out, layer3_out, layer4_out]
    

class ResNet18_layer1(nn.Module):
    def __init__(self):
        super(ResNet18_layer1, self).__init__()
        self.layer1 = resnet18().layer1
    
    def forward(self, x):
        return self.layer1(x)

class ResNet18_layer2(nn.Module):
    def __init__(self):
        super(ResNet18_layer2, self).__init__()
        self.layer2 = resnet18().layer2
    
    def forward(self, x):
        return self.layer2(x)
    
class ResNet18_layer3(nn.Module):
    def __init__(self):
        super(ResNet18_layer3, self).__init__()
        self.layer3 = resnet18().layer3
    
    def forward(self, x):
        return self.layer3(x)
    
class ResNet18_layer4(nn.Module):
    def __init__(self):
        super(ResNet18_layer4, self).__init__()
        self.layer4 = resnet18().layer4
    
    def forward(self, x):
        return self.layer4(x)
