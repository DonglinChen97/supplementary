import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from prune import *
from heapq import nsmallest
from operator import itemgetter
import numpy as np
import torch.nn.functional as F


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU(inplace=False)
        self.fc2   = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        raw = x
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        # return self.sigmoid(out)
        return self.relu1(self.sigmoid(out)*raw )

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.relu1 = nn.ReLU(inplace=False)#selu()
    def forward(self, x):
        raw = x
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        # return self.sigmoid(x)
        return self.relu1(self.sigmoid(x)*raw)


class Bottleneck(nn.Module):

    def __init__(self, inchannel, ratio=16, downsample=None):
        super(Bottleneck, self).__init__()

        self.ca = ChannelAttention(inchannel,ratio)
        self.sa = SpatialAttention()

        self.relu = nn.ReLU(inplace=False)

        self.downsample = downsample


    def forward(self, x):
        residual = x

        out = x

        out = self.ca(out) * out

        out = self.sa(out) * out
        

        if self.downsample is not None:
            residual = self.downsample(x)

        # out = out + residual
        out = self.relu(out)
        return out


class selu(nn.Module):
    def __init__(self):
        super(selu, self).__init__()
 
    def forward(self, x):
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        return scale * F.elu(x, alpha)



def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def blockUNet(in_c, out_c, name, transposed=0, bn=True, size=4, pad=1, dropout=0.):
    block = nn.Sequential()

    block.add_module('%s_relu' % name, nn.ReLU(inplace=True))

    if transposed == 0:
        block.add_module('%s_conv' % name, nn.Conv2d(in_c, out_c, kernel_size=size, stride=2, padding=pad, bias=True))
    elif transposed == 1:
        block.add_module('%s_upsam' % name, nn.Upsample(scale_factor=2))
        block.add_module('%s_tconv' % name, nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=pad, bias=True))
    else:
        block.add_module('%s_trans' % name, nn.ConvTranspose2d(in_c, out_c, (size, size), output_padding = 0,stride=2, padding=pad))
    if bn:
        block.add_module('%s_bn' % name, nn.BatchNorm2d(out_c))
    if dropout>0.:
        block.add_module('%s_dropout' % name, nn.Dropout2d( dropout, inplace=True))
    return block
    
# generator model
class TurbNetG(nn.Module):
    def __init__(self, channelExponent=6, dropout=0.):
        super(TurbNetG, self).__init__()
        channels = int(2 ** channelExponent + 0.5)

        self.layer1 = nn.Sequential()
        self.layer1.add_module('layer1_conv', nn.Conv2d(1, channels, 4, 2, 1, bias=True))
        self.layer1.add_module('layer1_bn', nn.BatchNorm2d(channels)) 

        self.CBAM1 = nn.Sequential()
        self.CBAM1.add_module('CBAM1_layer', SpatialAttention())

        self.layer2 = blockUNet(channels  , channels*2, 'layer2', transposed=0, bn=True, dropout=dropout )

        self.CBAM2 = nn.Sequential()
        self.CBAM2.add_module('CBAM2_layer', SpatialAttention())

        self.layer3= blockUNet(channels*2, channels*2, 'layer3',transposed=0, bn=True, dropout=dropout )

        self.CBAM3 = nn.Sequential()
        self.CBAM3.add_module('CBAM3_layer', SpatialAttention())

        self.layer4 = blockUNet(channels*2, channels*4, 'layer4', transposed=0, bn=True, dropout=dropout )

        self.CBAM4 = nn.Sequential()
        self.CBAM4.add_module('CBAM4_layer', SpatialAttention())

        self.layer5 = blockUNet(channels*4, channels*4, 'layer5', transposed=0, bn=True, dropout=dropout )

        self.CBAM5 = nn.Sequential()
        self.CBAM5.add_module('CBAM5_layer', SpatialAttention())

        self.layer6 = blockUNet(channels*4, channels*8, 'layer6', transposed=0, bn=True, dropout=dropout )

        self.CBAM6 = nn.Sequential()
        self.CBAM6.add_module('CBAM6_layer', SpatialAttention())#kernel_size=3

        self.layer7 = blockUNet(channels*8, channels*8, 'layer7', transposed=0, bn=True, dropout=dropout ,size=2, pad=0)

        self.CBAM7 = nn.Sequential()
        self.CBAM7.add_module('CBAM7_layer', ChannelAttention(channels*8,16))

        #self.dlayer7 = blockUNet(channels*8,channels*8, 'dlayer7', transposed=1, bn=True, dropout=dropout)
        self.dlayer7 = blockUNet(channels*8,channels*8, 'dlayer7', transposed=2, bn=False, dropout=dropout ,size=2, pad=0)
        self.dlayer6 = blockUNet(channels*16,channels*4, 'dlayer6', transposed=2, bn=False, dropout=dropout )
        self.dlayer5 = blockUNet(channels*8,channels*4, 'dlayer5', transposed=2, bn=False, dropout=dropout )
        self.dlayer4 = blockUNet(channels*8, channels*2, 'dlayer4', transposed=2, bn=False, dropout=dropout )
        self.dlayer3= blockUNet(channels*4, channels*2, 'dlayer3',transposed=2, bn=False, dropout=dropout )
        self.dlayer2 = blockUNet(channels*4, channels  , 'dlayer2', transposed=2, bn=False, dropout=dropout )
        self.dlayer1 = blockUNet(channels*2, 2  , 'dlayer1', transposed=2, bn=False, dropout=dropout )


    def forward(self, x):
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out1 = self.CBAM1(out1)

        out3 = self.layer3(out2)
        out2 = self.CBAM2(out2)

        out4 = self.layer4(out3)
        out3 = self.CBAM3(out3)

        out5 = self.layer5(out4)
        out4 = self.CBAM4(out4)

        out6 = self.layer6(out5)
        out5 = self.CBAM5(out5)

        out7 = self.layer7(out6)
        out6 = self.CBAM6(out6)
        # print (out6.shape)
        cbam_out = self.CBAM7(out7) + out7
        # cbam_out = torch.cat([out7, cbam_out], 1)

        dout7 = self.dlayer7(cbam_out)
        dout7_out6 = torch.cat([dout7, out6], 1)
        dout6 = self.dlayer6(dout7_out6)
        dout6_out5 = torch.cat([dout6, out5], 1)
        dout5 = self.dlayer5(dout6_out5)
        dout5_out4 = torch.cat([dout5, out4], 1)
        dout4 = self.dlayer4(dout5_out4)
        dout4_out3 = torch.cat([dout4, out3], 1)
        dout3 = self.dlayer3(dout4_out3)
        dout3_out2 = torch.cat([dout3, out2], 1)
        dout2 = self.dlayer2(dout3_out2)
        dout2_out1 = torch.cat([dout2, out1], 1)
        dout1 = self.dlayer1(dout2_out1)
        return dout1 
        




