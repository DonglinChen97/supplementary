import torch
import torch.nn as nn
import torch.nn.functional as F

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
# discriminator (only for adversarial training, currently unused)

def blockUNet(in_c, out_c, name, transposed=False, bn=True, relu=True, size=4, pad=0, stride=2):
    block = nn.Sequential()
    if relu:
        block.add_module('%s_relu' % name, nn.ReLU())
    else:
        block.add_module('%s_leakyrelu' % name, nn.LeakyReLU(0.2, inplace=True))
    if not transposed:
        block.add_module('%s_conv' % name, nn.Conv2d(in_c, out_c, kernel_size=size, stride=stride, padding=pad, bias=True))
    else:
        block.add_module('%s_tconv' % name, nn.ConvTranspose2d(in_c, out_c, kernel_size=size, stride=stride, padding=0, bias=True))
    # if bn:
    #     block.add_module('%s_bn' % name, nn.BatchNorm2d(out_c))

    return block

class TurbNetG(nn.Module):
    def __init__(self):
        super(TurbNetG, self).__init__()

        self.c0 = nn.Sequential()
        self.c0.add_module('layer1_conv', nn.Conv2d(1, 128 , (8,16), stride=(8,16), padding=0, bias=True))

        self.c1 = blockUNet(128  , 512, 'layer2', transposed=False, bn=True,  relu=True, size=4,stride=4, pad=0)
        
        # self.c1 = nn.Conv2d(128 , 512, 4, stride=4, padding=0)

        self.full = nn.Linear(8192,1024)

        self.d1_1 = blockUNet(1024  , 512, 'dlayer1_1', transposed=True, bn=True,  relu=True, size=(8,8),stride=(8,8), pad=0)
        # self.d1_1 = nn.ConvTranspose2d(1024, 512, (8,8), stride=(8,8), padding=0)
        self.d1_2 = blockUNet(512  , 256, 'dlayer1_2', transposed=True, bn=True,  relu=True, size=(4,8),stride=(4,8), pad=0)
        # self.d1_2 = nn.ConvTranspose2d(512, 256, (4,8), stride=(4,8), padding=0)
        self.d1_3 = blockUNet(256  , 32, 'dlayer1_3', transposed=True, bn=True,  relu=True, size=(2,2),stride=(2,2), pad=0)
        # self.d1_3 = nn.ConvTranspose2d(256, 32, (2,2), stride=2, padding=0)
        self.d1_4 = blockUNet(32  , 2, 'dlayer1_4', transposed=True, bn=False,  relu=True, size=(2,2),stride=(2,2), pad=0)
 

    def forward(self, x):
        raw = x 
        x = self.c0(x)

        x = self.c1(x)

        #b,c,_,_ = x.size()
        x = x.view(x.size(0),-1)
        out = self.full(x)
        out = out.view(out.size(0),out.size(1),1,1)


        d1 = self.d1_1(out)

        d1 = self.d1_2(d1)

        d1 = self.d1_3(d1)

        d1 = self.d1_4(d1)
        
        return d1

