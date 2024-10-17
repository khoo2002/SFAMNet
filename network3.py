import torch
import torch.nn as nn
import torch.nn.functional as F

# SFAMNet: A Scene Flow Attention-based Micro-expression Network
class Net_3D_MEAN(nn.Module):
    def _make_layer(self, block, planes, blocks, stride=1, att_type=None):
        inplanes = 1
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
            )
        layers = []
        layers.append(block(inplanes, planes, stride, downsample, use_TripletAttention=att_type=='TripletAttention'))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes, use_TripletAttention=att_type=='TripletAttention'))
        return nn.Sequential(*layers)

    def __init__(self, in_channels=1, out_channels=4):
        super(Net_3D_MEAN, self).__init__()
        self.c1 = self._make_layer(BasicBlock, 3,  1, att_type='TripletAttention')
        self.c2 = self._make_layer(BasicBlock, 3,  1, att_type='TripletAttention')
        self.c3 = self._make_layer(BasicBlock, 3,  1, att_type='TripletAttention')
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=3, padding=0)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avgpool2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.flatten = nn.Flatten()
        self.fc_spot = nn.Linear(in_features=392, out_features=1)
        self.fc_recog = nn.Linear(in_features=392, out_features=out_channels)
        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)
        self.sigmoid = nn.Sigmoid()
        self.conv1 = nn.Conv2d(9, out_channels=8, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(9, out_channels=8, kernel_size=5, padding=2)
        
    def forward(self, x1, x2, x3):
        x1 = self.c1(x1)
        x1 = self.maxpool1(x1)
        x2 = self.c2(x2)
        x2 = self.maxpool1(x2)
        x3 = self.c3(x3)
        x3 = self.maxpool1(x3)
        x = torch.cat((x1, x2, x3),1)
        x1 = self.conv1(x)
        x1 = self.maxpool2(x1)
        x1 = self.flatten(x1)
        x_spot = self.fc_spot(x1)
        x_spot = self.sigmoid(x_spot)
        x2 = self.conv2(x)
        x2 = self.maxpool2(x2)
        x2 = self.flatten(x2)
        x_recog = self.fc_recog(x2)
        return x_spot, x_recog

# CBAM
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat(
            (torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1
        )


class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(
            2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False
        )

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid_(x_out)
        return x * scale


class TripletAttention(nn.Module):
    def __init__(
        self,
        gate_channels,
        reduction_ratio=16,
        pool_types=["avg", "max"],
        no_spatial=False,
    ):
        super(TripletAttention, self).__init__()
        self.ChannelGateH = SpatialGate()
        self.ChannelGateW = SpatialGate()
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()

    def forward(self, x):
        x_perm1 = x.permute(0, 2, 1, 3).contiguous()
        x_out1 = self.ChannelGateH(x_perm1)
        x_out11 = x_out1.permute(0, 2, 1, 3).contiguous()
        x_perm2 = x.permute(0, 3, 2, 1).contiguous()
        x_out2 = self.ChannelGateW(x_perm2)
        x_out21 = x_out2.permute(0, 3, 2, 1).contiguous()
        if not self.no_spatial:
            x_out = self.SpatialGate(x)
            x_out = (1 / 3) * (x_out + x_out11 + x_out21)
        else:
            x_out = (1 / 2) * (x_out11 + x_out21)
        return x_out
        
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)
    
def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
    
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_TripletAttention=False):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

        if use_TripletAttention:
            self.TripletAttention = TripletAttention( planes, planes )
        else:
            self.TripletAttention = None

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        if not self.TripletAttention is None:
            out = self.TripletAttention(out)
        out += residual
        out = self.relu(out)
        return out
    
def conv5x5(in_planes, out_planes, stride=1):
    "5x5 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=5, stride=stride,
                     padding=2, bias=False)
    
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, use_TripletAttention=False):
        super(BasicBlock, self).__init__()
        self.conv1 = conv5x5(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv5x5(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        if use_TripletAttention:
            self.TripletAttention = TripletAttention( planes, planes )
        else:
            self.TripletAttention = None

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        if not self.TripletAttention is None:
            out = self.TripletAttention(out)
        out += residual
        out = self.relu(out)

        return out
