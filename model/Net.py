import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from model.ops_net import *
from util.ops_normal import *
from model.Trainer import ColorizationTrainer

class Net(nn.Module):
    def __init__(self, config):
        super(Net, self).__init__()
        self.config = config
        self.param = config.MODEL_PARAM
        self.trainer = None

        # Convolution block 1
        self.model_1 = nn.Sequential(*[ConvolutionModule(self.param.INPUT_DIM[0], 64, bn=False),
                                       ConvolutionModule(64, 64)])
        self.model_2 = nn.Sequential(*[ConvolutionModule(64, 128, bn=False),
                                       ConvolutionModule(128, 128)])
        self.model_3 = nn.Sequential(*[ConvolutionModule(128, 256),
                                       ConvolutionModule(128, 256, bn=False),
                                       ConvolutionModule(256, 256)])
        self.model_4 = nn.Sequential(*[ConvolutionModule(256, 512),
                                       ConvolutionModule(512, 512),
                                       ConvolutionModule(512, 512)])
        self.model_5 = nn.Sequential(*[ConvolutionModule(512, 512,dilation=2),
                                       ConvolutionModule(512, 512,dilation=2),
                                       ConvolutionModule(512, 512,dilation=2)])
        self.model_6 = nn.Sequential(*[ConvolutionModule(512, 512,dilation=2),
                                       ConvolutionModule(512, 512,dilation=2),
                                       ConvolutionModule(512, 512,dilation=2)])
        self.model_7 = nn.Sequential(*[ConvolutionModule(512, 512),
                                       ConvolutionModule(512, 512),
                                       ConvolutionModule(512, 512)])
        self.model_8_upsampling = nn.Sequential(
            *[nn.ConvTranspose2d(512,256,kernel_size=4, stride=2, padding=1, bias=False)]
        )
        self.short_cut_3_8 =nn.Sequential(*[ConvolutionModule(256,256,kernel_size=3, stride=1, padding=1, bn=False, relu=False)])

        self.model_8 = nn.Sequential(*[nn.ReLU(True),
                                        ConvolutionModule(256, 256, bn=False),
                                       ConvolutionModule(256, 256)])

        self.model_9_upsampling = nn.Sequential(
            *[nn.ConvTranspose2d(256,128,kernel_size=4, stride=2, padding=1, bias=False)]
        )
        self.short_cut_2_9 = nn.Sequential(*[ConvolutionModule(128,128,kernel_size=3, stride=1, padding=1, bn=False, relu=False)])

        self.model_9 = nn.Sequential(*[nn.ReLU(True),
                                       ConvolutionModule(128, 128)])

        self.model_10_upsampling = nn.Sequential(
            *[nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1, bias=False)]
        )
        self.short_cut_1_10 = nn.Sequential(*[ConvolutionModule(64, 128, kernel_size=3, stride=1, padding=1, bn=False, relu=False)])

        self.model_10 = nn.Sequential(*[nn.ReLU(True),
                                       ConvolutionModule(128, 128,bn=False,relu=False)])

        self.model_out = [ConvolutionModule(128,self.param.OUTPUT_DIM[0],kernel_size=1,padding=0,bn=False, relu=False)]
        self.model_out += [nn.Tanh()]

        self.model_out = nn.Sequential(*self.model_out)

    def forward(self, mono, hint, hint_mask):
        conv1_2 = self.model_1(torch.cat((mono, hint, hint_mask), dim=1))
        conv2_2 = self.model_2(conv1_2[:,:,::2,::2])
        conv3_3 = self.model_3(conv2_2[:,:,::2,::2])
        conv4_3 = self.model_4(conv3_3[:,:,::2,::2])
        conv5_3 = self.model_5(conv4_3)
        conv6_3 = self.model_6(conv5_3)
        conv7_3 = self.model_7(conv6_3)
        conv8_up = self.model_8_upsampling(conv7_3) + self.short_cut_3_8(conv3_3)
        conv8_3 = self.model_8(conv8_up)

        conv9_up = self.model_9_upsampling(conv8_3) + self.short_cut_2_9h(conv2_2)
        conv9_3 = self.model_9(conv9_up)
        conv10_up = self.model_10_upsampling(conv9_3) + self.short_cut_1_10(conv1_2)
        conv10_2 = self.model_10(conv10_up)
        res = self.model_out(conv10_2)

        return res

    def set_trainer(self, data_loader):
        self.trainer = ColorizationTrainer(self, data_loader)