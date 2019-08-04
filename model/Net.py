import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from model.ops_net import *
from shutil import copyfile
from util.ops_normal import *
from model.Trainer import ColorizationTrainer
from skimage.color import rgb2lab, lab2rgb
import os
import shutil
from tensorboardX import SummaryWriter


class BaseNet(nn.Module):
    def __init__(self, config, name):
        self.name = name

        super(BaseNet, self).__init__()
        self.config = config
        self.param = config.MODEL_PARAM
        self.trainer = None

    def keep_only_maximum_checkpoint(self, checkpoint_root, model_name):
        maximum = self.config.KEEP_LATEST_EPOCH
        filenames = os.listdir(checkpoint_root)
        file_list = []
        for filename in filenames:
            file_path = os.path.join(checkpoint_root, filename)
            if os.path.isdir(file_path) and filename.startswith(model_name):
                file_list.append(filename)
        file_list.sort()

        file_list = file_list[:-maximum]
        for target in file_list:
            shutil.rmtree(os.path.join(checkpoint_root, target))

    def save(self):
        model_name = get_original_model_name(self.config.MODEL_NAME)
        self.config.MODEL_NAME = model_name + "_{:04}".format(self.config['EPOCH'])

        checkpoint_folder = os.path.join(self.config.CHECKPOINT_PATH, self.config.MODEL_NAME)
        checkpoint_filename = "checkpoint.tar"
        config_filename = "net_info.yml"
        checkpoint_path = os.path.join(checkpoint_folder, checkpoint_filename)
        configuration_path = os.path.join(checkpoint_folder, config_filename)

        if not os.path.isfile(checkpoint_path):
            os.makedirs(checkpoint_folder, exist_ok=True)

        # config file change
        modify_yaml(self.config.CONFIG_PATH, configuration_path, self.config)
        self.config.CONFIG_PATH = configuration_path


        # save checkpoint
        save_item = dict()
        save_item[self.name] = self.state_dict()
        save_item[self.trainer.name] = self.trainer.optimizer.state_dict()

        torch.save(save_item, checkpoint_path)

        # clear checkpoint
        self.keep_only_maximum_checkpoint(self.config.CHECKPOINT_PATH, model_name)

    def load(self):
        checkpoint = None
        checkpoint_folder = os.path.join(self.config.CHECKPOINT_PATH, self.config.MODEL_NAME)
        checkpoint_filename = "checkpoint.tar"
        checkpoint_path = os.path.join(checkpoint_folder, checkpoint_filename)
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            self.load_state_dict(checkpoint[self.name], strict=True)
            if self.trainer is not None:
                self.trainer.optimizer.load_state_dict(checkpoint[self.trainer.name])
        else:
            print("[INFO] No checkpoint found at '{}'. check if it is first training.".format(checkpoint_path))

    def print_model_param(self):
        for param_tensor in self.state_dict():
            print(param_tensor, self.state_dict()[param_tensor].size())

    def set_trainer(self, data_loader):
        self.trainer = ColorizationTrainer(self, data_loader)


class ColorizationNet(BaseNet):
    """
        INPUT: mono[0,1], hint[0,1], hint mask[0,1]

    """

    def __init__(self, config, name='Colorization_network'):
        super(ColorizationNet, self).__init__(config, name)

        # Convolution block 1
        self.model_1 = nn.Sequential(*[ConvolutionModule(self.param.INPUT_DIM[0], 64, bn=False),
                                       ConvolutionModule(64, 64)])
        self.model_2 = nn.Sequential(*[ConvolutionModule(64, 128, bn=False),
                                       ConvolutionModule(128, 128)])
        self.model_3 = nn.Sequential(*[ConvolutionModule(128, 256),
                                       ConvolutionModule(256, 256, bn=False),
                                       ConvolutionModule(256, 256)])
        self.model_4 = nn.Sequential(*[ConvolutionModule(256, 512),
                                       ConvolutionModule(512, 512),
                                       ConvolutionModule(512, 512)])
        self.model_5 = nn.Sequential(*[ConvolutionModule(512, 512, padding=2, dilation=2),
                                       ConvolutionModule(512, 512, padding=2, dilation=2),
                                       ConvolutionModule(512, 512, padding=2,dilation=2)])
        self.model_6 = nn.Sequential(*[ConvolutionModule(512, 512, padding=2,dilation=2),
                                       ConvolutionModule(512, 512, padding=2,dilation=2),
                                       ConvolutionModule(512, 512, padding=2,dilation=2)])
        self.model_7 = nn.Sequential(*[ConvolutionModule(512, 512),
                                       ConvolutionModule(512, 512),
                                       ConvolutionModule(512, 512)])
        self.model_8_upsampling = nn.Sequential(
            *[nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False)]
        )
        self.short_cut_3_8 = nn.Sequential(
            *[ConvolutionModule(256, 256, kernel_size=3, stride=1, padding=1, bn=False, relu=False)])

        self.model_8 = nn.Sequential(*[nn.ReLU(True),
                                       ConvolutionModule(256, 256, bn=False),
                                       ConvolutionModule(256, 256)])

        self.model_9_upsampling = nn.Sequential(
            *[nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False)]
        )
        self.short_cut_2_9 = nn.Sequential(
            *[ConvolutionModule(128, 128, kernel_size=3, stride=1, padding=1, bn=False, relu=False)])

        self.model_9 = nn.Sequential(*[nn.ReLU(True),
                                       ConvolutionModule(128, 128)])

        self.model_10_upsampling = nn.Sequential(
            *[nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1, bias=False)]
        )
        self.short_cut_1_10 = nn.Sequential(
            *[ConvolutionModule(64, 128, kernel_size=3, stride=1, padding=1, bn=False, relu=False)])

        self.model_10 = nn.Sequential(*[nn.ReLU(True),
                                        ConvolutionModule(128, 128, bn=False, relu=False)])

        self.model_out = [
            ConvolutionModule(128, self.param.OUTPUT_DIM[0], kernel_size=1, padding=0, bn=False, relu=False)]
        self.model_out += [nn.Tanh()]

        self.model_out = nn.Sequential(*self.model_out)

    def forward(self, l, ab, hint_mask):
        l, ab = self.preprocessing(l, ab, hint_mask)

        conv1_2 = self.model_1(torch.cat((l, ab, hint_mask), dim=1))
        conv2_2 = self.model_2(conv1_2[:, :, ::2, ::2])
        conv3_3 = self.model_3(conv2_2[:, :, ::2, ::2])
        conv4_3 = self.model_4(conv3_3[:, :, ::2, ::2])
        conv5_3 = self.model_5(conv4_3)
        conv6_3 = self.model_6(conv5_3)
        conv7_3 = self.model_7(conv6_3)
        conv8_up = self.model_8_upsampling(conv7_3) + self.short_cut_3_8(conv3_3)
        conv8_3 = self.model_8(conv8_up)

        conv9_up = self.model_9_upsampling(conv8_3) + self.short_cut_2_9(conv2_2)
        conv9_3 = self.model_9(conv9_up)
        conv10_up = self.model_10_upsampling(conv9_3) + self.short_cut_1_10(conv1_2)
        conv10_2 = self.model_10(conv10_up)
        res = self.model_out(conv10_2)

        return res

    @staticmethod
    def preprocessing(luma, ab, hint_mask):
        with torch.no_grad():
            luma -= 50
            luma /= 50
            ab /= 100
            ab *= hint_mask
        return luma, ab


    @staticmethod
    def lab_normal(lab):
        """
        :param lab: torch tensor
        :return: normalize lab
        """
        dim = len(lab.size())
        if dim == 4:
            lab[:, :1, :, :] = (lab[:, :1, :, :] - 50) / 50
            lab[:, 1:, :, :] = lab[:, 1:, :, :] / 100
        elif dim == 3:
            lab[:1, :, :] = (lab[:1, :, :] - 50) / 50
            lab[1:, :, :] = lab[1:, :, :] / 100
        else:
            Exception()
        return lab

    @staticmethod
    def lab_unnormal(lab):
        dim = len(lab.size())
        if dim == 4:
            lab[:, :1, :, :] = lab[:, :1, :, :] * 50 + 50
            lab[:, 1:, :, :] = lab[:, 1:, :, :] * 100
        elif dim == 3:
            lab[:1, :, :] = lab[:1, :, :] * 50 + 50
            lab[1:, :, :] = lab[1:, :, :] * 100
        else:
            Exception()

        return lab
