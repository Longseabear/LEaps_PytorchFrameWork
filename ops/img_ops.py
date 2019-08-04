import skimage.color
import torch
import torchvision as tv

def img_numpy_to_tensor(img, shape_transform=False):
    if img.ndim==3:
        img = img.transpose((2,0,1))
    else:
        img = img.transpose((0,3,1,2))
    return torch.from_numpy(img).float()

def img_tensor_to_numpy(img, shape_transform=False):
    if img.ndimension()==3:
        img = img.permute((1,2,0))
    else:
        img = img.permute((0,2,3,1))
    return img.numpy()
