import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F
from torchvision.models import resnet18


class Channel_Attention(nn.Module):
    def __init__(self, in_ch, reduction=4):
        super(Channel_Attention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        mip = max(8, in_ch // reduction)
        self.fc1 = nn.Conv2d(in_ch, mip, 1, bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(2*mip, in_ch, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        k = x
        avg = self.avg_pool(x)
        max = self.max_pool(x)
        avg_ = self.relu1(self.fc1(avg))
        max_ = self.relu1(self.fc1(max))

        x_ch = torch.cat([avg_, max_], dim=1)

        out = self.fc2(x_ch)
        out = self.sigmoid(out)
        out = k * out


        return out


class Spatial_Attention(nn.Module):
    def __init__(self, kernel_size=7):
        super(Spatial_Attention, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)

        attn = self.sigmoid(self.conv(x_cat))
        out = x * attn
        return out
    

class CABM(nn.Module):
    def __init__(self, in_ch, reduction=4, kernel_size=7):
        super(CABM, self).__init__()
        self.channel_attention = Channel_Attention(in_ch, reduction)
        self.spatial_attention = Spatial_Attention(kernel_size)

    def forward(self, x):
        out = self.channel_attention(x)
        out = self.spatial_attention(out)
        return out


class Double_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


# To align the wideth and height of each feature map
class Align(nn.Module):
    def __init__(self, channels=[64, 128, 256, 512], in_channel=None, out_channel=None, cat=True, position_embedding=[]):
        super(Align, self).__init__()
        self.cat = cat
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.position_embedding = position_embedding

        self.channels = channels
        self.cabm0 = CABM(self.channels[0])
        self.cabm1 = CABM(self.channels[1])
        self.cabm2 = CABM(self.channels[2])
        self.cabm3 = CABM(self.channels[3])

        # self.cabm2_0 = CABM(self.channels[0])
        # self.cabm2_1 = CABM(self.channels[1])
        # self.cabm2_2 = CABM(self.channels[2])
        # self.cabm2_3 = CABM(self.channels[3])

        if in_channel and out_channel:
            self.final_conv = Double_conv(in_channel, out_channel)
            # self.final_conv2 = Double_conv(in_channel, out_channel)

    def down(self, x, n):
        nh, nw = n, n
        return rearrange(x, 'b c (nh h) (nw w) -> b (c nh nw) h w', nh=nh, nw=nw)
    
    def up(self, x, ratio, mode='bilinear'):
        return(F.interpolate(x, scale_factor=ratio, mode=mode))

    def align(self, x, ratio):
        if ratio > 1:
            return self.up(x, ratio)
        elif ratio < 1:
            return self.down(x, int(1/ratio))
        else:
            return x

    def forward(self, T1, T2, layer):
        scales = [1, 2, 4, 8, 16]
        ratios = [scales[i] / scales[layer] for i in range(len(scales))]

        T1[0] = self.cabm0(T1[0])
        T1[1] = self.cabm1(T1[1])
        T1[2] = self.cabm2(T1[2])
        T1[3] = self.cabm3(T1[3])

        T2[0] = self.cabm0(T2[0])
        T2[1] = self.cabm1(T2[1])
        T2[2] = self.cabm2(T2[2])
        T2[3] = self.cabm3(T2[3])

        if len(self.position_embedding) == len(self.channels):
            T1[0] = T1[0] + self.position_embedding[0]
            T1[1] = T1[1] + self.position_embedding[1]
            T1[2] = T1[2] + self.position_embedding[2]
            T1[3] = T1[3] + self.position_embedding[3]

            T2[0] = T2[0] + self.position_embedding[0]
            T2[1] = T2[1] + self.position_embedding[1]
            T2[2] = T2[2] + self.position_embedding[2]
            T2[3] = T2[3] + self.position_embedding[3]
            # print("position enbedded")


        if self.cat:
            T1_merged = torch.cat([self.align(f, ratios[i]) for i, f in enumerate(T1)], dim=1)
            T2_merged = torch.cat([self.align(f, ratios[i]) for i, f in enumerate(T2)], dim=1)
        else:
            T1_merged = [self.align(f, ratios[i]) for i, f in enumerate(T1)]
            T2_merged = [self.align(f, ratios[i]) for i, f in enumerate(T2)]

        if self.in_channel and self.out_channel:
            T1_merged = self.final_conv(T1_merged)
            T2_merged = self.final_conv(T2_merged)

        return T1_merged, T2_merged


# ResNet18 backbone
class Encoder(nn.Module):
    def __init__(self, pretrained=False):
        super(Encoder, self).__init__()
        basic_resnet = resnet18(weights='IMAGENET1K_V1' if pretrained else None)

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
    

# 4 layers of ResNet18
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

