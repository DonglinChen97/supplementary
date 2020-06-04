import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from prune import *
from heapq import nsmallest
from operator import itemgetter
import numpy as np
import torch.nn.functional as F



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

    #block.add_module('%s_relu' % name, nn.ReLU(inplace=True))
    #block.add_module('%s_leakyrelu' % name, nn.LeakyReLU(0.2, inplace=True))
    block.add_module('%s_elu' % name, nn.ELU())
    if transposed == 0:
        block.add_module('%s_conv' % name, nn.Conv2d(in_c, out_c, kernel_size=size, stride=2, padding=pad, bias=True))
    elif transposed == 1:
        block.add_module('%s_upsam' % name, nn.Upsample(scale_factor=2))
        #block.add_module('%s_tconv' % name, nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=pad, bias=True))
    else:
        block.add_module('%s_trans' % name, nn.ConvTranspose2d(in_c, out_c, (size, size), output_padding = 0,stride=2, padding=pad))



    if bn:
        #block.add_module('%s_bn' % name, nn.GroupNorm(16,out_c))
        block.add_module('%s_bn' % name, nn.BatchNorm2d(out_c))
        #block.add_module('%s_bn' % name, nn.InstanceNorm2d(out_c))
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

        self.layer2 = blockUNet(channels  , channels*2, 'layer2', transposed=0, bn=True, dropout=dropout )
        self.layer3= blockUNet(channels*2, channels*2, 'layer3',transposed=0, bn=True, dropout=dropout )
        self.layer4 = blockUNet(channels*2, channels*4, 'layer4', transposed=0, bn=True, dropout=dropout )
        self.layer5 = blockUNet(channels*4, channels*4, 'layer5', transposed=0, bn=True, dropout=dropout )
        self.layer6 = blockUNet(channels*4, channels*8, 'layer6', transposed=0, bn=True, dropout=dropout )
        self.layer7 = blockUNet(channels*8, channels*8, 'layer7', transposed=0, bn=True, dropout=dropout ,size=2, pad=0)

        #self.dlayer7 = blockUNet(channels*8,channels*8, 'dlayer7', transposed=1, bn=True, dropout=dropout )
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
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)
        out6 = self.layer6(out5)
        out7 = self.layer7(out6)


        dout7 = self.dlayer7(out7)
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
    

