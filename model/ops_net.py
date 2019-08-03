import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvolutionModule(nn.Module):
    num = 1

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, deconv=False, bn=True, relu=True,**kwargs):
        super(ConvolutionModule, self).__init__()

        self.name = "ConvolutionModule_{}".format(ConvolutionModule.num)
        ConvolutionModule.num += 1

        self.relu = relu
        self.use_bn = bn
        self.deconv = deconv
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.stride = stride

        if deconv:
            self.conv = nn.ConvTranspose2d(in_channels, out_channels, bias=False, **kwargs)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels,kernel_size=kernel_size,stride=stride, padding=padding, dilation=dilation, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

    # def __str__(self):
    #     str = '-------------------------------------------------\n'
    #     str += 'Module Name = [{}]\n'.format(self.name)
    #     if self.deconv:
    #         str += '[B,H,W,{}]->[B,H*2,W*2,{}]\n'.format(self.in_channels,self.out_channels)
    #     else:
    #         str += '[B,H,W,{}]->[B,H,W,{}]\n'.format(self.in_channels,self.out_channels)
    #     str+='Option:  dilate {} / deconv {} / bn {} / relu {}\n'.format(self.dilation,self.deconv,self.bn,self.relu)
    #     str+='         kernel {}X{} / Stride {}\n'.format(self.kernel_size,self.kernel_size, self.stride)
    #     str+='-------------------------------------------------\n'
    #     return str
