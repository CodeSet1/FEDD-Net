import math
import torch
from torch import nn
import torch.nn.functional as F
from inspect import isfunction


def exists(x):      #检查变量x是否存在
    return x is not None


def default(val, d):        #如果val存在，则返回val；否则，如果d是一个函数，则调用d()返回其结果，否则直接返回d
    if exists(val):
        return val
    return d() if isfunction(d) else d

###########生成位置编码，在处理序列数据时非常有用###########
# PositionalEncoding Source： https://github.com/lmnt-com/wavegrad/blob/master/src/wavegrad/model.py
class PositionalEncoding(nn.Module):        
    def __init__(self, dim):        #初始化编码的维度
        super().__init__()
        self.dim = dim

    def forward(self, noise_level):
        count = self.dim // 2   #计算步长，维度的一半
        step = torch.arange(count, dtype=noise_level.dtype,
                            device=noise_level.device) / count
        encoding = noise_level.unsqueeze(
            1) * torch.exp(-math.log(1e4) * step.unsqueeze(0))      
        encoding = torch.cat(
            [torch.sin(encoding), torch.cos(encoding)], dim=-1)     #使用正弦和余弦函数生成的编码，最后将它们拼接在一起
        return encoding

###########对输入特征进行特征级放射变换###########
class FeatureWiseAffine(nn.Module):
    def __init__(self, in_channels, out_channels, use_affine_level=False):      #输入通道数、输出通道数和是否使用仿射级别
        super(FeatureWiseAffine, self).__init__()
        self.use_affine_level = use_affine_level        #根据use_affine_level参数决定是否进行仿射变换。如果使用仿射级别，则对输入特征进行缩放和平移；否则，仅进行平移
        self.noise_func = nn.Sequential(
            nn.Linear(in_channels, out_channels*(1+self.use_affine_level))
        )

    def forward(self, x, noise_embed):      #向输入x添加噪声，噪声通过noise_func函数生成的，并且可以以仿射变换的形式添加到输入中
        batch = x.shape[0]      #获取输入x的batch大小
        if self.use_affine_level:       #检查是否使用仿射变换
            gamma, beta = self.noise_func(noise_embed).view(
                batch, -1, 1, 1).chunk(2, dim=1)        #将噪声嵌入noise_embed通过noise_func生成噪声，并将其塑造为batch个样本，每个样本有-1个通道（即噪声维度），分成两个部分gamma和beta
            x = (1 + gamma) * x + beta      #将gamma和beta应用到输入x上，实现仿射变换
        else:
            x = x + self.noise_func(noise_embed).view(batch, -1, 1, 1)      #如果不使用仿射变换，直接将噪声添加到输入x上
        return x

##########Swish激活函数##########
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

##########上采样和下采样操作##########
class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")       #使用最近邻插值方法将输入x放大两倍
        self.conv = nn.Conv2d(dim, dim, 3, padding=1)       #使用卷积层对上采样的结果进行卷积操作，保持特征图的尺寸不变

    def forward(self, x):
        return self.conv(self.up(x))


class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, 2, 1)        #使用卷积层对输入x进行卷积操作，步幅为2，保持特征图的通道数不变，但将尺寸减半

    def forward(self, x):
        return self.conv(x)


# building block modules


class Block(nn.Module):     #基本卷即块
    def __init__(self, dim, dim_out, groups=32, dropout=0):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(groups, dim),      #对输入进行组归一化，以减少内部协变量偏移
            Swish(),        #一种自门控激活函数
            nn.Dropout(dropout) if dropout != 0 else nn.Identity(),     #如果dropout不为0，则使用Dropout层，否则使用Identity层，即不进行任何操作
            nn.Conv2d(dim, dim_out, 3, padding=1)       #使用3x3的卷积核进行卷积操作，填充1，保持特征图的尺寸不变
        )

    def forward(self, x):
        return self.block(x)


class ResnetBlock(nn.Module):       #残差块，包含Block和一个残差连接。主要用于保持输入和输出的维度一致，从而在深层网络中保持信息流动。
    def __init__(self, dim, dim_out, noise_level_emb_dim=None, dropout=0, use_affine_level=False, norm_groups=32):
        super().__init__()
        self.noise_func = FeatureWiseAffine(        
            noise_level_emb_dim, dim_out, use_affine_level)

        self.block1 = Block(dim, dim_out, groups=norm_groups)
        self.block2 = Block(dim_out, dim_out, groups=norm_groups, dropout=dropout)
        self.res_conv = nn.Conv2d(
            dim, dim_out, 1) if dim != dim_out else nn.Identity()      

    def forward(self, x, time_emb):
        b, c, h, w = x.shape
        h = self.block1(x)
        h = self.noise_func(h, time_emb)        #根据时间嵌入time_emb调整特征
        h = self.block2(h)
        return h + self.res_conv(x)      #残差连接：如果输入和输出的维度不一致，使用1*1卷积进行维度调整


class SelfAttention(nn.Module):     #自注意力机制，用于捕捉输入特征之间的依赖关系
    def __init__(self, in_channel, n_head=1, norm_groups=32):       #输入特征的通道数，注意力头的数量，默认为1，组归一化的组数=32
        super().__init__()

        self.n_head = n_head        #存储注意力头的数量

        self.norm = nn.GroupNorm(norm_groups, in_channel)       #对输入进行组归一化
        self.qkv = nn.Conv2d(in_channel, in_channel * 3, 1, bias=False)     #生成qkv
        self.out = nn.Conv2d(in_channel, in_channel, 1)     #用于输出注意力机制的最终结果

    def forward(self, input):
        batch, channel, height, width = input.shape 
        n_head = self.n_head        #获取注意力头的数量
        head_dim = channel // n_head     #计算每个注意力头的维度

        norm = self.norm(input)      #对输入进行归一化
        qkv = self.qkv(norm).view(batch, n_head, head_dim * 3, height, width)       #通过卷积层生成查询、键和值，并调整其形状
        query, key, value = qkv.chunk(3, dim=2)  # bhdyx    将qkv分成查询、键和值

        # 计算注意力权重
        attn = torch.einsum(
            "bnchw, bncyx -> bnhwyx", query, key
        ).contiguous() / math.sqrt(channel)     #计算查询和键的点积，得到注意力权重
        attn = attn.view(batch, n_head, height, width, -1)      
        attn = torch.softmax(attn, -1)      #归一化注意力权重，使用softmax函数
        attn = attn.view(batch, n_head, height, width, height, width)     # 调整注意力权重的形状

        # 计算输出
        out = torch.einsum("bnhwyx, bncyx -> bnchw", attn, value).contiguous()      #使用注意力权重和值计算输出，contiguous()确保张量在内存中是连续存储的
        out = self.out(out.view(batch, channel, height, width))     #通过卷积层调整输出形状

        return out + input      #将原始输入与注意力机制的输出相加，实现残差连接，有助于梯度的反向传播


class ResnetBlocWithAttn(nn.Module):    #带有自注意力机制的残差块
    def __init__(self, dim, dim_out, *, noise_level_emb_dim=None, norm_groups=32, dropout=0, with_attn=False):  #norm_groups=32,归一化组数
        super().__init__()
        self.with_attn = with_attn  #保存了是否使用注意力机制的标志
        self.res_block = ResnetBlock(   #残差块
            dim, dim_out, noise_level_emb_dim, norm_groups=norm_groups, dropout=dropout)
        if with_attn:
            self.attn = SelfAttention(dim_out, norm_groups=norm_groups)

    def forward(self, x, time_emb): 
        x = self.res_block(x, time_emb)     #时间嵌入time_emb，用于在时间序列数据中引入时间信息
        if(self.with_attn):
            x = self.attn(x)
        return x


class UNet(nn.Module):      #UNet架构
    def __init__(
        self,
        in_channel=6,
        out_channel=3,
        inner_channel=32,
        norm_groups=32,
        channel_mults=(1, 2, 4, 8, 8),
        attn_res=(8),
        res_blocks=3,
        dropout=0,
        with_noise_level_emb=True,
        image_size=128
    ):
        super().__init__()

        if with_noise_level_emb:        #如果with_noise_level_emb为True，则创建一个噪声级别嵌入层noise_level_mlp，用于处理时间信息
            noise_level_channel = inner_channel
            self.noise_level_mlp = nn.Sequential(
                PositionalEncoding(inner_channel),
                nn.Linear(inner_channel, inner_channel * 4),
                Swish(),
                nn.Linear(inner_channel * 4, inner_channel)
            )
        else:
            noise_level_channel = None
            self.noise_level_mlp = None
#下采样路径
        num_mults = len(channel_mults)  #定义通道数的倍增因子  num_mults=5
        pre_channel = inner_channel
        feat_channels = [pre_channel]
        now_res = image_size
        downs = [nn.Conv2d(in_channel, inner_channel,
                           kernel_size=3, padding=1)]
        for ind in range(num_mults):
            is_last = (ind == num_mults - 1)
            use_attn = (now_res in attn_res)        #检查now_res是否在attn_re
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks):
                downs.append(ResnetBlocWithAttn(
                    pre_channel, channel_mult, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups, dropout=dropout, with_attn=use_attn))
                feat_channels.append(channel_mult)
                pre_channel = channel_mult
            if not is_last:
                downs.append(Downsample(pre_channel))
                feat_channels.append(pre_channel)
                now_res = now_res//2
        self.downs = nn.ModuleList(downs)
#中间层
        self.mid = nn.ModuleList([
            ResnetBlocWithAttn(pre_channel, pre_channel, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups,
                               dropout=dropout, with_attn=True),
            ResnetBlocWithAttn(pre_channel, pre_channel, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups,
                               dropout=dropout, with_attn=False)
        ])      #mid列表存储中间层，包括两个ResNet块，一个使用注意力机制，另一个不使用
#上采样路径
        ups = []    #初始化上采样路径，包括上采样层和ResNet块，ups列表存储上采样路径中的所有层
        for ind in reversed(range(num_mults)):
            is_last = (ind < 1)
            use_attn = (now_res in attn_res)
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks+1):
                ups.append(ResnetBlocWithAttn(
                    pre_channel+feat_channels.pop(), channel_mult, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups,
                        dropout=dropout, with_attn=use_attn))
                pre_channel = channel_mult
            if not is_last:
                ups.append(Upsample(pre_channel))
                now_res = now_res*2    #now_res = image_size 

        self.ups = nn.ModuleList(ups)
#最终卷积层
        self.final_conv = Block(pre_channel, default(out_channel, in_channel), groups=norm_groups)  #final_conv是最终的卷积层，用于将特征图转换为输出

    def forward(self, x, time):
        t = self.noise_level_mlp(time) if exists(
            self.noise_level_mlp) else None

        feats = []
        for layer in self.downs:    #通过下采样路径，将输入图像转换为特征图
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(x, t)
            else:
                x = layer(x)
            feats.append(x)

        for layer in self.mid:  #通过中间层处理特征图
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(x, t)
            else:
                x = layer(x)

        for layer in self.ups:  #通过上采样路径，将特征图恢复到原始图像尺寸
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(torch.cat((x, feats.pop()), dim=1), t)
            else:
                x = layer(x)

        return self.final_conv(x)   #最后将特征图转换为输出

class consist_Unet(nn.Module):   #自编码器架构consist_Unet，通过编解码结构来提取特征并重建图像
    def __init__(
        self,
        in_channel=3,
        out_channel=3,
        inner_channel=32,
        norm_groups=32,
        channel_mults=(1, 2, 4, 8, 8),
        attn_res=(8),
        res_blocks=3,
        dropout=0,
        with_noise_level_emb=True,
        image_size=128
    ):
        super().__init__()

        if with_noise_level_emb:
            noise_level_channel = inner_channel
            self.noise_level_mlp = nn.Sequential(
                PositionalEncoding(inner_channel),
                nn.Linear(inner_channel, inner_channel * 4),
                Swish(),
                nn.Linear(inner_channel * 4, inner_channel)
            )
        else:
            noise_level_channel = None
            self.noise_level_mlp = None

        num_mults = len(channel_mults)
#编码器：多个卷积层和下采样层组成，用于逐步降低图像的分辨率并提取特征
        pre_channel = inner_channel
        feat_channels = [pre_channel]
        now_res = image_size
        downs = [nn.Conv2d(in_channel, inner_channel,
                           kernel_size=3, padding=1)]
        for ind in range(num_mults):
            is_last = (ind == num_mults - 1)
            use_attn = (now_res in attn_res)        #检查now_res是否在attn_res中，以决定是否使用注意力机制
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks):
                downs.append(ResnetBlocWithAttn(
                    pre_channel, channel_mult, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups, dropout=dropout, with_attn=use_attn))
                feat_channels.append(channel_mult)
                pre_channel = channel_mult
            if not is_last:
                downs.append(Downsample(pre_channel))
                feat_channels.append(pre_channel)
                now_res = now_res//2
        self.downs = nn.ModuleList(downs)
#中间层：两个残差块，一个使用注意力机制，另一个不使用
        self.mid = nn.ModuleList([
            ResnetBlocWithAttn(pre_channel, pre_channel, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups,
                               dropout=dropout, with_attn=True),
            ResnetBlocWithAttn(pre_channel, pre_channel, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups,
                               dropout=dropout, with_attn=False)
        ])
#解码器部分：由多个上采样层和残差块组成，用于逐步恢复图像的分辨率
        ups = []
        for ind in reversed(range(num_mults)):
            is_last = (ind < 1)
            use_attn = (now_res in attn_res)
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks+1):
                ups.append(ResnetBlocWithAttn(
                    pre_channel+feat_channels.pop(), channel_mult, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups,
                        dropout=dropout, with_attn=use_attn))
                pre_channel = channel_mult
            if not is_last:
                ups.append(Upsample(pre_channel))
                now_res = now_res*2

        self.ups = nn.ModuleList(ups)
#最终卷积层将解码器输出的特征图转换为输出图像
        self.final_conv = Block(pre_channel, default(out_channel, in_channel), groups=norm_groups)

    def forward(self, x, time):
        x_res = x
        t = self.noise_level_mlp(time) if exists(
            self.noise_level_mlp) else None

        feats = []
        for layer in self.downs:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(x, t)
            else:
                x = layer(x)
            feats.append(x)

        for layer in self.mid:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(x, t)
            else:
                x = layer(x)

        for layer in self.ups:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(torch.cat((x, feats.pop()), dim=1), t)        #将当前特征图和编码器部分提取的特征图进行拼接，然后通过残差块进行特征融合
            else:
                x = layer(x)        #feats.pop()弹出一个元素，即从feats列表中删除并返回最后一个元素

        return self.final_conv(x) + x_res
