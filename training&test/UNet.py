import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from prune import *
from heapq import nsmallest
from operator import itemgetter
import numpy as np
import torch.nn.functional as F

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class TurbNetG(nn.Module):
    def __init__(self, channelExponent=6, dropout=0.):
        super(TurbNetG, self).__init__()
        channels = int(2 ** channelExponent + 0.5)

        self.pool = nn.MaxPool2d(2)

        self.layer1 = nn.Sequential()
        self.layer1.add_module('layer1_conv1', nn.Conv2d(1, channels, 3, 1, 1, bias=True))
        self.layer1.add_module('layer1_relu1', nn.ReLU(inplace=True))
        self.layer1.add_module('layer1_conv2', nn.Conv2d(channels, channels, 3, 1, 1, bias=True))
        self.layer1.add_module('layer1_relu2', nn.ReLU(inplace=True))
        
 

        self.layer2 = nn.Sequential()
        self.layer2.add_module('layer2_conv1', nn.Conv2d(channels, channels*2, 3, 1, 1, bias=True))
        self.layer2.add_module('layer2_relu1', nn.ReLU(inplace=True))
        self.layer2.add_module('layer2_conv2', nn.Conv2d(channels*2, channels*2, 3, 1, 1, bias=True))
        self.layer2.add_module('layer2_relu2', nn.ReLU(inplace=True))
        # self.layer2.add_module('layer2_pool', nn.MaxPool2d(2))

        self.layer3 = nn.Sequential()
        self.layer3.add_module('layer3_conv1', nn.Conv2d(channels*2, channels*4, 3, 1, 1, bias=True))
        self.layer3.add_module('layer3_relu1', nn.ReLU(inplace=True))
        self.layer3.add_module('layer3_conv2', nn.Conv2d(channels*4, channels*4, 3, 1, 1, bias=True))
        self.layer3.add_module('layer3_relu2', nn.ReLU(inplace=True))
        # self.layer3.add_module('layer3_pool', nn.MaxPool2d(2))

        self.layer4 = nn.Sequential()
        self.layer4.add_module('layer4_conv1', nn.Conv2d(channels*4, channels*8, 3, 1, 1, bias=True))
        self.layer4.add_module('layer4_relu1', nn.ReLU(inplace=True))
        self.layer4.add_module('layer4_conv2', nn.Conv2d(channels*8, channels*8, 3, 1, 1, bias=True))
        self.layer4.add_module('layer4_relu2', nn.ReLU(inplace=True))
        # self.layer4.add_module('layer4_pool', nn.MaxPool2d(2))

        self.dlayer1 = nn.Sequential()
        self.dlayer1.add_module('dlayer1_conv1', nn.Conv2d(channels*8, channels*16, 3, 1, 1, bias=True))
        self.dlayer1.add_module('dlayer1_relu1', nn.ReLU(inplace=True))
        self.dlayer1.add_module('dlayer1_conv2', nn.Conv2d(channels*16, channels*16, 3, 1, 1, bias=True))
        self.dlayer1.add_module('dlayer1_relu2', nn.ReLU(inplace=True))
        self.dlayer1.add_module('dlayer1_upsam' , nn.Upsample(scale_factor=2))
        self.dlayer1.add_module('dlayer1_tconv' , nn.Conv2d(channels*16, channels*8,  3, 1, 1, bias=True))

        self.dlayer2 = nn.Sequential()
        self.dlayer2.add_module('dlayer2_conv1', nn.Conv2d(channels*16, channels*8, 3, 1, 1, bias=True))
        self.dlayer2.add_module('dlayer2_relu1', nn.ReLU(inplace=True))
        self.dlayer2.add_module('dlayer2_conv2', nn.Conv2d(channels*8, channels*8, 3, 1, 1, bias=True))
        self.dlayer2.add_module('dlayer2_relu2', nn.ReLU(inplace=True))
        self.dlayer2.add_module('dlayer2_upsam' , nn.Upsample(scale_factor=2))
        self.dlayer2.add_module('dlayer2_tconv' , nn.Conv2d(channels*8, channels*4,  3, 1, 1, bias=True))

        self.dlayer3 = nn.Sequential()
        self.dlayer3.add_module('dlayer3_conv1', nn.Conv2d(channels*8, channels*4, 3, 1, 1, bias=True))
        self.dlayer3.add_module('dlayer3_relu1', nn.ReLU(inplace=True))
        self.dlayer3.add_module('dlayer3_conv2', nn.Conv2d(channels*4, channels*4, 3, 1, 1, bias=True))
        self.dlayer3.add_module('dlayer3_relu2', nn.ReLU(inplace=True))
        self.dlayer3.add_module('dlayer3_upsam' , nn.Upsample(scale_factor=2))
        self.dlayer3.add_module('dlayer3_tconv' , nn.Conv2d(channels*4, channels*2,  3, 1, 1, bias=True))

        self.dlayer4 = nn.Sequential()
        self.dlayer4.add_module('dlayer4_conv1', nn.Conv2d(channels*4, channels*2, 3, 1, 1, bias=True))
        self.dlayer4.add_module('dlayer4_relu1', nn.ReLU(inplace=True))
        self.dlayer4.add_module('dlayer4_conv2', nn.Conv2d(channels*2, channels*2, 3, 1, 1, bias=True))
        self.dlayer4.add_module('dlayer4_relu2', nn.ReLU(inplace=True))
        self.dlayer4.add_module('dlayer4_upsam' , nn.Upsample(scale_factor=2))
        self.dlayer4.add_module('dlayer4_tconv' , nn.Conv2d(channels*2, channels,  3, 1, 1, bias=True))

        self.dlayer5 = nn.Sequential()
        self.dlayer5.add_module('dlayer5_conv1', nn.Conv2d(channels*2, channels, 3, 1, 1, bias=True))
        self.dlayer5.add_module('dlayer5_relu1', nn.ReLU(inplace=True))
        self.dlayer5.add_module('dlayer5_conv2', nn.Conv2d(channels, channels, 3, 1, 1, bias=True))
        self.dlayer5.add_module('dlayer5_relu2', nn.ReLU(inplace=True))
        # self.dlayer5.add_module('dlayer5_upsam' , nn.Upsample(scale_factor=2))
        self.dlayer5.add_module('dlayer5_tconv' , nn.Conv2d(channels, 2,  1, 1, 0, bias=True))

    def forward(self, x):
        out1 = self.layer1(x)
        temp = self.pool(out1)
        out2 = self.layer2(temp)
        temp = self.pool(out2)
        out3 = self.layer3(temp)
        temp = self.pool(out3)
        out4 = self.layer4(temp)
        temp = self.pool(out4)

        dout1 = self.dlayer1(temp)
        dout1_out4 = torch.cat([dout1, out4], 1)
        dout2 = self.dlayer2(dout1_out4)
        dout2_out3 = torch.cat([dout2, out3], 1)
        dout3 = self.dlayer3(dout2_out3)
        dout3_out2 = torch.cat([dout3, out2], 1)
        dout4 = self.dlayer4(dout3_out2)
        dout4_out1 = torch.cat([dout4, out1], 1)
        dout5 = self.dlayer5(dout4_out1)

        return dout5
    

