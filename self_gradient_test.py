import torch
from torch.autograd import function
from torch.nn import *
import time
import shutil
import os
from tensorboardX import SummaryWriter
from model.Net import ColorizationNet
from ops.img_ops import *
from model.Net import ColorizationNet
from skimage import io
from skimage.color import rgb2lab, lab2rgb
import numpy as np
#
# img = io.imread('Lenna.png')
# img = img[:,:255,:3]
# img = np.stack([img,]*8
#                ,axis=0)
# print(img.shape)
# img = rgb2lab(img)
# print(img.shape, img[0]==img[1])
# img = lab2rgb(img)
# print(img.shape)
#
# input()
# print(img[2,1].min(), 'orign')
# img = img_numpy_to_tensor(img)
# img = ColorizationNet.lab_normal(img)
# img = img_tensor_to_numpy(img)
# print(img[2,1].min(),'normal')
# img = img_numpy_to_tensor(img)
# img = ColorizationNet.lab_unnormal(img)
# img = img_tensor_to_numpy(img)
# print(img[2,1].min(),'unnormal')

# print(img.shape, img.ndim)
#
# new_img = img_numpy_to_tensor(img)
# print(new_img, new_img.shape)
#
new_img = torch.rand([3,5,5])
new_img = img_tensor_to_numpy(new_img)
print(new_img.shape)
# print(new_img, new_img.shape)

#
# writer3 = SummaryWriter('summary/O/EPOCH_1', comment='3x learning rate',)
# writer2 = SummaryWriter('summary/O/EPOCH_2', comment='3x learning rate',)
#
# MODEL_NAME = "colorizationNet_+afakgfag_145r1"
# if str.rfind(MODEL_NAME,'_') is not -1:
#     print(MODEL_NAME[:str.rfind(MODEL_NAME,'_')])
# else:
#     print(MODEL_NAME)

torch.manual_seed(32)
b = torch.ones([2,3,4,4])
a = b.cuda()
print(a.is_cuda, b.is_cuda)

# a =[1,2,3,4,5]
# print(a[:-1],a[1:])
# print(a[:-2],a[2:])
#
# print(a)
# a = torch.rand([1,32,256,256],dtype=torch.float).cuda()
# b = torch.rand([1,32,256,256],dtype=torch.float).cuda()
#
# x = Conv3d(32,32,(198,3,3)).cuda()
#
# stime = time.time()
# c = a*b
# etime = time.time()
# print(c.size(),etime-stime)
#
# stime = time.time()
# b = x(a)
# etime = time.time()
# print(b.size(),etime-stime)
#
# x = Conv3d(32,32,3).cuda()
# stime = time.time()
# x(a)
# etime = time.time()
# print(etime-stime)
