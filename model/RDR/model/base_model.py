import os
import torch
import torch.nn as nn


class BaseModel():
    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device(
            'cuda' if opt['gpu_ids'] is not None else 'cpu')
        self.begin_step = 0
        self.begin_epoch = 0

#######数据加载#######
    def feed_data(self, data):
        pass

#######参数优化#######
    def optimize_parameters(self): 
        pass

#######获取当前可视化结果#######
    def get_current_visuals(self):
        pass

#######获取当前损失#######
    def get_current_losses(self):
        pass

#######打印网络结构#######
    def print_network(self):
        pass

#######设置设备#######
    def set_device(self, x):
        if isinstance(x, dict):
            for key, item in x.items():
                if item is not None:
                    x[key] = item.to(self.device)
        elif isinstance(x, list):
            for item in x:
                if item is not None:
                    item = item.to(self.device)
        else:
            x = x.to(self.device)
        return x

#######获取网络描述#######
    def get_network_description(self, network):
        '''Get the string and total parameters of the network'''
        if isinstance(network, nn.DataParallel):
            network = network.module
        s = str(network)        #str(network)获取网络的结构描述
        n = sum(map(lambda x: x.numel(), network.parameters()))     #计算网络的总参数数量
        return s, n     #返回网络的结构描述和参数总数
