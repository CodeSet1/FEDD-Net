import numpy as np
import random

import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F


def pad_if_smaller(img, size, fill=0):    #对图像进行填充，使其尺寸至少达到指定的最小的尺寸，如果图像尺寸已经大于或等于指定的最小尺寸，则不进行任何操作。
    min_size = min(img.size)            
    if min_size < size:
        ow, oh = img.size
        padh = size - oh if oh < size else 0
        padw = size - ow if ow < size else 0    
        img = F.pad(img, (0, 0, padw, padh), fill=fill)
    return img      


class Compose(object):      #将多个图像预处理操作组合成一个整体
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class Resize(object):       #调整图像和目标的大小
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = F.resize(image, self.size)
        target = F.resize(target, self.size, interpolation=T.InterpolationMode.NEAREST)
        return image, target


class RandomHorizontalFlip(object):     #以一定的概率水平翻转图像和目标
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            target = F.hflip(target)
        return image, target


class RandomVerticalFlip(object):       #以一定的概率垂直翻转图像和目标
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = F.vflip(image)
            target = F.vflip(target)
        return image, target


class RandomCrop(object):               #从图像和目标中随机裁剪出一个指定大小的区域
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = pad_if_smaller(image, self.size)
        target = pad_if_smaller(target, self.size)
        crop_params = T.RandomCrop.get_params(image, (self.size, self.size))
        image = F.crop(image, *crop_params)
        target = F.crop(target, *crop_params)
        return image, target

class CenterCrop(object):               #从图像和目标中裁剪出一个居中指定大小的区域
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = F.center_crop(image, self.size)
        target = F.center_crop(target, self.size)
        return image, target


class ToTensor(object):                 #将图像和目标转换为张量格式
    def __call__(self, image, target):
        image = F.to_tensor(image)
        target = F.to_tensor(target)
        return image, target