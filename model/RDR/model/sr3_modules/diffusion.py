import math
import torch
from torch import device, nn, einsum
import torch.nn.functional as F
from inspect import isfunction
from functools import partial
import numpy as np
from tqdm import tqdm

########用于生成不同的噪声调度（beta schedule）#########
def _warmup_beta(linear_start, linear_end, n_timestep, warmup_frac):        #生成一个带有预热(warmup)阶段的beta序列，起始值，结束值，时间步数，预热阶段占总时间步数的比例
    betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    warmup_time = int(n_timestep * warmup_frac)
    betas[:warmup_time] = np.linspace(
        linear_start, linear_end, warmup_time, dtype=np.float64)
    return betas

########根据指定的调度类型生成beta序列#########
def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    if schedule == 'quad':
        betas = np.linspace(linear_start ** 0.5, linear_end ** 0.5,
                            n_timestep, dtype=np.float64) ** 2
    elif schedule == 'linear':
        betas = np.linspace(linear_start, linear_end,
                            n_timestep, dtype=np.float64)
    elif schedule == 'warmup10':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.1)
    elif schedule == 'warmup50':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.5)
    elif schedule == 'const':    # 生成一个全为linear_end的beta序列
        betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    elif schedule == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1  生成一个递减的beta序列
        betas = 1. / np.linspace(n_timestep,
                                 1, n_timestep, dtype=np.float64)
    elif schedule == "cosine":      #使用余弦函数生成beta序列
        timesteps = (       #生成时间步长序列，生成了一个从0到1的时间步长序列，每个时间不长都除以总时间步长n_timestep，然后加上cosine_s。目的是为了调整余弦函数的形状
            torch.arange(n_timestep + 1, dtype=torch.float64) /
            n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * math.pi / 2       #计算alph序列
        alphas = torch.cos(alphas).pow(2)       #取平方
        alphas = alphas / alphas[0]     #将序列归一化，使其第一个元素为1
        betas = 1 - alphas[1:] / alphas[:-1]        #计算beta序列
        betas = betas.clamp(max=0.999)      #使用clamp函数将beta序列中的值限制在0到0.999之间，以避免数值不稳定
    else:
        raise NotImplementedError(schedule)
    return betas


# gaussian diffusion trainer class

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        denoise_fn,     #去噪函数，用于生成图像
        Unet_fn,        #UNet模型，用于图像处理
        image_size,
        channels=3,
        loss_type='l1',
        conditional=True,       #是否使用条件扩散，默认为True
        schedule_opt=None       #噪声调度选项，默认为None
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.denoise_fn = denoise_fn
        self.Unet_fn = Unet_fn
        self.loss_type = loss_type
        self.conditional = conditional
        self.ddim_timesteps = 5
        self.ddim_eta = 1.0
        if schedule_opt is not None:
            pass
            # self.set_new_noise_schedule(schedule_opt)

    def set_loss(self, device):     #设置损失函数
        if self.loss_type == 'l1':      #使用L1损失函数（绝对误差）
            self.loss_func = nn.L1Loss(reduction='sum').to(device)
        elif self.loss_type == 'l2':        #使用L2损失函数（均方误差）
            self.loss_func = nn.MSELoss(reduction='sum').to(device)
        else:
            raise NotImplementedError()

    def set_new_noise_schedule(self, schedule_opt, device):
        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)        #使用partial函数创建一个部分应用函数to_torch,用与将数据转换为指定数据类型和设备上的张量

        betas = make_beta_schedule(     #计算beta
            schedule=schedule_opt['schedule'],
            n_timestep=schedule_opt['n_timestep'],
            linear_start=schedule_opt['linear_start'],
            linear_end=schedule_opt['linear_end'])
        betas = betas.detach().cpu().numpy() if isinstance(
            betas, torch.Tensor) else betas
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)     #alpha的累积乘积
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])        #alphas 累积乘积的前一个值
        self.sqrt_alphas_cumprod_prev = np.sqrt(        #sqrt_alphas_cumprod_prev的平方根
            np.append(1., alphas_cumprod))

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev',
                             to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others  前向过程
        self.register_buffer('sqrt_alphas_cumprod',
                             to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod',
                             to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod',
                             to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)  计算反向过程所需的函数
        posterior_variance = betas * \
            (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)      
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance',      
                             to_torch(posterior_variance))      #后验分布的方差
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(
            np.log(np.maximum(posterior_variance, 1e-20))))     #后验分布的方差的对数，并进行了裁剪，防止出现负数
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))      #后验分布的均值的系数
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))      #后验分布的均值的系数

    def predict_start_from_noise(self, x_t, t, noise):      #从噪声图像x_t和时间步t中预测初始图像x_start
        return self.sqrt_recip_alphas_cumprod[t] * x_t - \
            self.sqrt_recipm1_alphas_cumprod[t] * noise     #通过与计算的累积平方根倒数和平方根倒数减1的积累值，来计算初始图像x_start

    def q_posterior(self, x_start, x_t, t):     #计算给定的初始图像、噪声图像和时间步时的后验均值和后验对数方差
        posterior_mean = self.posterior_mean_coef1[t] * \
            x_start + self.posterior_mean_coef2[t] * x_t
        posterior_log_variance_clipped = self.posterior_log_variance_clipped[t]
        return posterior_mean, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool, condition_x=None):     #计算给定噪声图像 x、时间步 t、是否裁剪重建图像的标志 clip_denoised 以及条件图像 condition_x 时的模型均值、后验对数方差、重建图像和去噪重建图像。
        batch_size = x.shape[0]
        noise_level = torch.FloatTensor(
            [self.sqrt_alphas_cumprod_prev[t+1]]).repeat(batch_size, 1).to(x.device)        #计算噪声水平
        if condition_x is not None:     #如果有条件图像condition_x，则使用 predict_start_from_noise 方法结合去噪函数 denoise_fn 和条件图像来预测重建图像 x_recon。
            x_recon = self.predict_start_from_noise(
                x, t=t, noise=self.denoise_fn(torch.cat([condition_x, x], dim=1), noise_level))
        else:
            x_recon = self.predict_start_from_noise(
                x, t=t, noise=self.denoise_fn(x, noise_level))

        if clip_denoised:       #如果需要裁剪，则对重建图像x_recon进行裁剪
            x_recon.clamp_(-1., 1.)
        x_denoise_recon = x_recon

        x_recon = self.Unet_fn(torch.cat([condition_x, x], dim=1), noise_level)     
        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_log_variance, x_recon, x_denoise_recon

##########从给定的噪声样本x和时间步t中生成一个样本##########
    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True, condition_x=None):
        model_mean, model_log_variance, x_0, x_denoise_recon = self.p_mean_variance(
            x=x, t=t, clip_denoised=clip_denoised, condition_x=condition_x)     #获取模型预测的均值，对数方差，去噪后的样本和去噪后的重建样本
        noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)       #生成与x形状相同的噪声noise，如果t>0则生成标准正态噪声，否则生成全零噪声
        return model_mean + noise * (0.5 * model_log_variance).exp(), x_0, x_denoise_recon      #生成的样本，x0去噪后的样本，去噪后的重建样本

###########通过多次调用p_sample生成一系列样本，通过逐步减少噪声来生成样本，直到达到指定的步数##########
    @torch.no_grad()
    def p_sample_loop(self, x_in, continous=False):     #x_in为初始噪声样本，continous为是否返回所有生成的样本，还是只返回最后一个样本
        device = self.betas.device
        sample_inter = (1 | (self.num_timesteps//10))       #定义采样间隔sample_inter,每隔sample_inter步保存一次样本
        if not self.conditional:        #如果不是条件模型，则初始化样本img为标准正态噪声，并循环调用p_sample函数生成样本
            shape = x_in
            img = torch.randn(shape, device=device)
            ret_img = img
            ret_x_0 = img
            ret_x_0_denoise = img
            for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
                img, x_0, x_0_denoise = self.p_sample(img, i)
                if i % sample_inter == 0:
                    ret_img = torch.cat([ret_img, img], dim=0)
                    ret_x_0 = torch.cat([ret_x_0, x_0], dim=0)
                    ret_x_0_denoise = torch.cat([ret_x_0_denoise, x_0_denoise], dim=0)
        else:
            x = x_in        #如果是条件模型，则初始化样本img为x_in，并循环调用p_sample函数生成样本,每次调用时传入condition_x
            shape = x.shape
            img = torch.randn(shape, device=device)
            ret_img = x
            ret_x_0 = x
            ret_x_0_denoise = x
            for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
                img, x_0, x_0_denoise = self.p_sample(img, i, condition_x=x)
                if i % sample_inter == 0:
                    ret_img = torch.cat([ret_img, img], dim=0)
                    ret_x_0 = torch.cat([ret_x_0, x_0], dim=0)
                    ret_x_0_denoise = torch.cat([ret_x_0_denoise, x_0_denoise], dim=0)      #每隔sample_inter步,将生成的样本、去噪后的样本和去噪后的重建样本保存到ret_img、ret_x_0、ret_x_0_denoise中
        if continous:
            return ret_img, ret_x_0, ret_x_0_denoise
        else:
            return ret_img[-1], ret_x_0, ret_x_0_denoise

###########使用DDIM算法进行图像超分辨率处理###########
    @torch.no_grad()
    def super_resolution(self, x_in, continous=False):
        return self.p_sample_loop(x_in, continous)      #调用p_sample_loop函数进行超分辨率处理

    @torch.no_grad()
    def super_resolution_ddim(self, x_in, continous=False):
        return self.p_sample_loop_ddim(x_in, continous)     #调用p_sample_loop_ddim函数进行超分辨率处理

###########DDIM算法的核心步骤，用于生成超分辨率图像###########
    @torch.no_grad()
    def p_sample_loop_ddim(self, x_in, continous=False):
        device = self.betas.device
        ddim_timesteps = self.ddim_timesteps        #DDIM算法的时间步长
        interval = self.num_timesteps // ddim_timesteps     #时间步长的间隔
        timestep_seq = np.arange(self.num_timesteps - 1, -1, -interval)[::-1]       #时间步长序列，从num_timesteps-1开始到0，步长为interval，倒序排列
        prev_timestep_seq = np.append(np.array([-1]), timestep_seq[:-1])        #前一时间步长序列

        sample_interval = (1 | (ddim_timesteps // 10))
        x = x_in
        shape = x.shape
        b, c, h, w = shape

        img = torch.randn(shape, device=device)
        ret_img = x
        x0_img = x
        denoise_x0_img = x

        for i in tqdm(reversed(range(ddim_timesteps)), desc='Sampling loop Diff-RDA time step', total=ddim_timesteps):
            t = torch.tensor([timestep_seq[i]] * b, dtype=torch.long, device=device)
            prev_t = torch.tensor([prev_timestep_seq[i]] * b, dtype=torch.long, device=device)

            alpha_cumprod_t = self.extract_from_tensor(self.alphas_cumprod, t, img.shape)
            alpha_cumprod_t_prev = torch.ones_like(alpha_cumprod_t) if i == 0 else self.extract_from_tensor(self.alphas_cumprod,
                                                                                                 prev_t, img.shape)     #在每个时间步长上，计算当前和前一个时间步长的累积概率
            noise_level = torch.full((b, 1), self.sqrt_alphas_cumprod_prev[t + 1], device=x.device, dtype=torch.float32)        #计算噪声水平

            pred_noise = self.denoise_fn(torch.cat([x_in, img], dim=1), noise_level)        #使用denoise_fn函数预测噪声pred_noise
            x_recon = self.predict_start_from_noise(img, t=t, noise=pred_noise)     #使用predict_start_from_noise函数预测初始图像x_recon
            x_recon.clamp_(-1., 1.)
            denoise_x0 = x_recon

            pred_x0 = self.Unet_fn(torch.cat([x_in, x_recon], dim=1), noise_level)      #使用Unet_fn函数预测当前时间步长的图像
            pred_x0.clamp_(-1., 1.)

            sigmas_t = self.ddim_eta * torch.sqrt(
                (1 - alpha_cumprod_t_prev) / (1 - alpha_cumprod_t) * (1 - alpha_cumprod_t / alpha_cumprod_t_prev))      #计算当前时间步长的噪声标准差sigmas_t

            x_t_prev = torch.sqrt(alpha_cumprod_t_prev) * pred_x0 + torch.sqrt(1 - alpha_cumprod_t_prev - sigmas_t ** 2) * pred_noise \
                       + sigmas_t * torch.randn_like(pred_x0)
            img = x_t_prev      #根据DDIM算法更新图像img

            if i % sample_interval == 0:
                ret_img = torch.cat([ret_img, img], dim=0)
                x0_img = torch.cat([x0_img, x_recon], dim=0)
                denoise_x0_img = torch.cat([denoise_x0_img, denoise_x0], dim=0)     #每隔一定时间步长，将当前图像、预测的初始图像和去噪后的初始图象添加到返回结果中

        return (ret_img, x0_img, denoise_x0_img) if continous else (ret_img[-1], x0_img, denoise_x0_img)

    def extract_from_tensor(self, alpha, t, x_shape):       #从alpha张量中提取特定时间步t的值
        b = t.shape[0]
        out = torch.index_select(alpha.to(t.device), 0, t).float()      #使用torch.index_select函数从alpha张量中提取特定时间步t的值，然后调整形状以匹配输入张量x_shape
        out = out.view(b, *([1] * (len(x_shape) - 1)))
        return out      #在扩散模型中，alpha是累积的平方根系数，用于控制噪声的强度

    def q_sample(self, x_start, continuous_sqrt_alpha_cumprod, noise=None):     #根据当前时间步的continious_sqrt_alpha_cumprod和初始图像x_start生成带噪声的图像
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            continuous_sqrt_alpha_cumprod * x_start +
            (1 - continuous_sqrt_alpha_cumprod**2).sqrt() * noise
        )

#########计算预测噪声和真是噪声之间的损失##########
    def p_losses(self, x_in, noise=None):
        x_start = x_in['high']      #从x_in中提取'high'键对应的值作为x_start，即原始图像
        [b, c, h, w] = x_start.shape
        t = np.random.randint(1, self.num_timesteps + 1)        #从1到self.num_timesteps之间随机选择一个整数作为时间步t，表示在扩散过程中的时间步
        continuous_sqrt_alpha_cumprod = torch.FloatTensor(      #使用np.random.uniform在self.sqrt_alphas_cumprod_prev[t-1]和self.sqrt_alphas_cumprod_prev[t]之间生成随机值，这些值用于控制噪声的添加量。
            np.random.uniform(
                self.sqrt_alphas_cumprod_prev[t-1],
                self.sqrt_alphas_cumprod_prev[t],
                size=b
            )
        ).to(x_start.device)
        continuous_sqrt_alpha_cumprod = continuous_sqrt_alpha_cumprod.view(
            b, -1)              #view方法接受一个或多个整数作为参数，这些整数指定了新的形状

        noise = default(noise, lambda: torch.randn_like(x_start))       #如果未提供noise，则生成与x_start形状相同的随机噪声
        x_noisy = self.q_sample(
            x_start=x_start, continuous_sqrt_alpha_cumprod=continuous_sqrt_alpha_cumprod.view(-1, 1, 1, 1), noise=noise)        #使用self.q_sample方法根据x_start、continuous_sqrt_alpha_cumprod和noise生成加噪后的图像x_noisy

        if not self.conditional:
            pred_noise = self.denoise_fn(x_noisy, continuous_sqrt_alpha_cumprod)        #如果模型不是条件模型，则仅使用x_noisy作为输入预测噪声
        else:
            pred_noise = self.denoise_fn(torch.cat([x_in['low'], x_noisy], dim=1), continuous_sqrt_alpha_cumprod)       #如果是条件模型，则将x_in['low']和x_noisy拼接在一起作为输入预测噪声

        x_0_from_denoise = self.predict_start_from_noise(x_noisy, t=t-1, noise=pred_noise)      
        x_0_from_denoise.clamp_(-1., 1.)        #限制张量x_0_from_denoise的范围为[-1, 1]，使得所有小于-1的值变为-1，所有大于1的值变为1，而介于-1和1之间的值保持不变

        #x_0 = self.Unet_fn(torch.cat([x_in['low'], x_0_from_denoise], dim=1), continuous_sqrt_alpha_cumprod)
        x_0 = self.Unet_fn(torch.cat([x_in['low'], x_0_from_denoise.detach()], dim=1), continuous_sqrt_alpha_cumprod)       #将x_in['low'](条件信息)和x_0_from_denoise（去噪后的图像）拼接在一起，通过self.Unet_fn得到预测初始图像x_0
        x_0.clamp_(-1., 1.)

###########计算损失##########
        loss_x0 = self.loss_func(x_0, x_start)      #l1损失
        loss_eps = self.loss_func(noise, pred_noise)        #l1损失

        loss = loss_eps + loss_x0       #总损失
        return loss, loss_eps, loss_x0      #返回总损失

    def forward(self, x, *args, **kwargs):      #任意数量的位置参数（*args）和关键字参数（**kwargs）
        return self.p_losses(x, *args, **kwargs)
