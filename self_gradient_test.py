import torch
from torch.autograd import function
from torch.nn import *
import time

torch.manual_seed(32)
a = torch.rand([2,2,2])
print(a)

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
