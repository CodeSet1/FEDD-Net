import logging
from collections import OrderedDict

import torch
import torch.nn as nn
import os
try:
    import model.networks as networks
except ImportError:
    import model.networks as networks

from .base_model import BaseModel
logger = logging.getLogger('base')


class DDPM(BaseModel):      #定义一个DDPM类，继承自BaseModel类
    def __init__(self, opt):
        super(DDPM, self).__init__(opt)
        # define network and load pretrained models 网络定义和加载与训练模型
        self.netG = self.set_device(networks.define_G(opt))
        self.schedule_phase = None      #初始化为None,用于后续的噪声调度
        
        # set loss and load resume state    设置损失函数和加载恢复状态
        self.set_loss()
        self.set_new_noise_schedule(        #用于设置新的噪声调度，这里使用的是训练阶段的噪声调度
            opt['model']['beta_schedule']['train'], schedule_phase='train')
        if self.opt['phase'] == 'train':        #训练阶段设置
            self.netG.train()
            # find the parameters to optimize
            if opt['model']['finetune_norm']:       #如果配置中指定了微调归一化层opt['model']['finetune_norm']，则只对包含“Transformer”的参数进行优化，并将这些参数初始化为0，并将这些参数加入到优化器中
                optim_params = []
                for k, v in self.netG.named_parameters():
                    v.requires_grad = False
                    if k.find('transformer') >= 0:
                        v.requires_grad = True
                        v.data.zero_()
                        optim_params.append(v)
                        logger.info(
                            'Params [{:s}] initialized to 0 and will optimize.'.format(k))
            else:
                optim_params = list(self.netG.parameters())     #否则，对所有参数进行优化

            self.optG = torch.optim.AdamW(      #创建优化器，使用AdamW算法，学习率为opt['train']["optimizer"]["lr"]
                optim_params, lr=opt['train']["optimizer"]["lr"])
            self.log_dict = OrderedDict()   #用于存储训练过程中的日志信息
        self.load_network()     #加载与训练模型或恢复训练状态
        self.print_network()    #打印网络结构，方便调试和记录

    def feed_data(self, data):      
        self.data = self.set_device(data)

    def optimize_parameters(self):      #优化参数
        self.optG.zero_grad()       #归零优化器（self.optG）的梯度，防止梯度累积
        l_pix, loss_eps, loss_x0 = self.netG(self.data)     #计算损失函数
        # need to average in multi-gpu
        b, c, h, w = self.data['high'].shape        #
        l_pix = l_pix.sum()/int(b*c*h*w)    #计算每个损失函数的综合并除以总像素数
        loss_eps = loss_eps.sum()/int(b*c*h*w)
        loss_x0 = loss_x0.sum()/int(b*c*h*w)
        l_pix.backward()    #损失函数进行反向传播，计算梯度
        self.optG.step()    #更新模型的参数

        # set log
        self.log_dict['l_pix'] = l_pix.item()   
        self.log_dict['loss_eps'] = loss_eps.item()
        self.log_dict['loss_x0'] = loss_x0.item()

    def test(self, continous=False):    
        self.netG.eval()    #将模型设置为评估模式
        with torch.no_grad():   #禁用梯度计算，以节省内存和计算资源
            if isinstance(self.netG, nn.DataParallel):  #进行多GPU并行处理，调用不同的超分辨率方法
                self.SR, self.x_0, self.x_0_denoise = self.netG.module.super_resolution(
                    self.data['low'], continous)
            else:
                self.SR, self.x_0, self.x_0_denoise = self.netG.super_resolution(
                    self.data['low'], continous)
        self.netG.train()   #在推断完成后，这行代码将模型恢复到训练模式，这是为了确保如果后续需要训练模型，模型能够正确地应用Dropout和BatchNorm等层。

    def test_ddim(self, continous=False):
        self.netG.eval()
        with torch.no_grad():
            if isinstance(self.netG, nn.DataParallel):
                self.SR, self.x_0, self.x_0_denoise = self.netG.module.super_resolution_ddim(   #这里的self.netG.module是因为nn.DataParallel会将其包装的模型封装在一个额外的层中，因此需要通过.module来访问实际的模型层。
                    self.data['low'], continous)
            else:
                self.SR, self.x_0, self.x_0_denoise = self.netG.super_resolution_ddim(
                    self.data['low'], continous)
        self.netG.train()

    def set_loss(self):     #设置损失函数
        if isinstance(self.netG, nn.DataParallel):      #根据self.netG是否是nn.DataParallel的实例，选择调用self.netG.module.set_loss(self.device)或self.netG.set_loss(self.device)。
            self.netG.module.set_loss(self.device)
        else:
            self.netG.set_loss(self.device)

    def set_new_noise_schedule(self, schedule_opt, schedule_phase='train'):     #设置新的噪声调度
        if self.schedule_phase is None or self.schedule_phase != schedule_phase:
            self.schedule_phase = schedule_phase
            if isinstance(self.netG, nn.DataParallel):
                self.netG.module.set_new_noise_schedule(        #在训练过程中，根据不同的训练阶段设置不同的噪声调度
                    schedule_opt, self.device)
            else:
                self.netG.set_new_noise_schedule(schedule_opt, self.device)

    def get_current_log(self):      #获取当前日志信息
        return self.log_dict

    def get_current_visuals(self, sample=False):        #获取当前的可视化结果
        out_dict = OrderedDict()        #根据simple参数的值，返回不同的可视化结果。如果sample为True，则返回采样结果，否则返回包括原始图像、低分辨率图像、高分辨率图像等在内的多个可视化结果
        if sample:
            out_dict['SAM'] = self.SR.detach().float().cpu()
        else:
            out_dict['DDPM'] = self.SR.detach().float().cpu()
            out_dict['x_0'] = self.x_0.detach().float().cpu()
            out_dict['x_0_denoise'] = self.x_0_denoise.detach().float().cpu()
            out_dict['low'] = self.data['low'].detach().float().cpu()
            out_dict['high'] = self.data['high'].detach().float().cpu()
        return out_dict

    def get_current_visuals_val(self, sample=False):        #获取当前验证过程中的可视化结果，但只返回高分辨率图像的采样结果。
        out_dict = OrderedDict()
        out_dict['DDPM'] = self.SR.detach().float().cpu()       #确保self.SR在验证过程中被正确更新
        return out_dict

    def print_network(self):        #打印网络结构
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)

        logger.info(
            'Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
        logger.info(s)

    def save_network(self, epoch, iter_step, best_psnr_flag=False):     #保存网络结构
        if best_psnr_flag == True:      #根据best_psnr_flag的值，选择保存路径。将self.netG的状态字典保存到指定路径，并保存优化器的状态。
            gen_path = os.path.join(
                self.opt['path']['checkpoint'], 'best_gen.pth')
            opt_path = os.path.join(
                self.opt['path']['checkpoint'], 'best_opt.pth')
        else:
            gen_path = os.path.join(
                self.opt['path']['checkpoint'], 'Finall_gen.pth')
            opt_path = os.path.join(
                self.opt['path']['checkpoint'], 'Finall_opt.pth')
        # gen
        network = self.netG     #获取生成器网络
        if isinstance(self.netG, nn.DataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, gen_path)
        # opt  创建优化器状态字典
        opt_state = {'epoch': epoch, 'iter': iter_step,
                     'scheduler': None, 'optimizer': None}
        opt_state['optimizer'] = self.optG.state_dict()
        torch.save(opt_state, opt_path)

        logger.info(
            'Saved model in [{:s}] ...'.format(gen_path))       #记录日志：使用日志记录器记录模型保存路径

    def load_network(self):     #获取加载路径
        load_path = self.opt['path']['resume_state']
        if load_path is not None:
            logger.info(
                'Loading pretrained model for G [{:s}] ...'.format(load_path))
            gen_path = '{}_gen.pth'.format(load_path)
            # gen
            network = self.netG
            if isinstance(self.netG, nn.DataParallel):
                network = network.module
            network.load_state_dict(torch.load(
                gen_path), strict=(not self.opt['model']['finetune_norm']))
            if self.opt['phase'] == 'train':
                # optimizer
                opt_path = '{}_opt.pth'.format(load_path)
                opt = torch.load(opt_path)
                #self.optG.load_state_dict(opt['optimizer'])
                self.begin_step = opt['iter']
                self.begin_epoch = opt['epoch']
