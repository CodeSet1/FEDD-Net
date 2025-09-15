import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from einops import rearrange
import dtcwt
import numpy as np

class DecomNet(nn.Module):
    def __init__(self, inp_channel = 3, channel=32, kernel_size=3):     #inp_channel输入通道数；channel卷积核的通道数；kernel_size是卷积核的大小
        super(DecomNet, self).__init__()
        self.net1_conv0 = nn.Conv2d(inp_channel, channel, kernel_size, padding=1, padding_mode='replicate')     #inp_channel--->channel
        self.net1_convs = nn.Sequential(nn.Conv2d(channel, channel, 5, padding=2, padding_mode='replicate'),
                                        nn.ReLU(),
                                        nn.Conv2d(channel, channel, kernel_size, padding=1, padding_mode='replicate'),
                                        nn.ReLU(),
                                        nn.Conv2d(channel, channel, kernel_size, padding=1, padding_mode='replicate'),
                                        nn.ReLU())
        self.net1_recon = nn.Conv2d(channel, 1,  1, padding=0, padding_mode='replicate')        #将特征图转换为单通道输出

        self.TDN_R = TDN(dim=24)        #dim=24模块输入维度

    def forward(self, input_im):
        R = self.TDN_R(input_im)
        feats0 = self.net1_conv0(input_im)
        featss = self.net1_convs(feats0)
        outs = self.net1_recon(featss)
        R = torch.sigmoid(R)
        L = torch.sigmoid(outs)     #使用sigmoid激活函数将输出限制在[0,1]范围内
        return R, L
########################################################################## 
class DirectionalFilter(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(DirectionalFilter, self).__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=(kernel_size-1)//2, bias=False)

        # 初始化滤波器权重为Gabor滤波器
        self.init_gabor_filter()

    def init_gabor_filter(self):
        # Gabor滤波器参数
        sigma = 1.0
        theta = np.pi / 4  # 45度方向
        lambd = 10.0
        psi = np.pi / 2
        gabor_kernel = self.gabor_kernel((self.kernel_size, self.kernel_size), theta, lambd, psi, sigma)

        # 将Gabor滤波器转换为PyTorch张量并初始化滤波器权重
        gabor_weights = torch.tensor(gabor_kernel, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        self.conv.weight.data = gabor_weights.repeat(self.conv.out_channels, self.conv.in_channels, 1, 1)  # (out_channels, in_channels, H, W)

    def gabor_kernel(self, size, theta, lambd, psi, sigma):
        # Gabor滤波器核函数
        x = np.arange(0, size[0], 1, float)
        y = np.arange(0, size[1], 1, float)
        x, y = np.meshgrid(x, y)

        x -= size[0] // 2
        y -= size[1] // 2

        rot_x = x * np.cos(theta) + y * np.sin(theta)
        rot_y = -x * np.sin(theta) + y * np.cos(theta)

        gabor = np.exp(-.5 * ((rot_x ** 2 + rot_y ** 2) / sigma ** 2))
        gabor *= np.cos(2 * np.pi * (rot_x / lambd + psi))

        return gabor

class DirectionalAttention(nn.Module):
    def __init__(self, channels):
        super(DirectionalAttention, self).__init__()
        self.query_conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.key_conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.value_conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.scale = (channels ** -0.5)

    def forward(self, x):
        B, C, H, W = x.shape
        q = self.query_conv(x).reshape(B, C, -1).permute(0, 2, 1)
        k = self.key_conv(x).reshape(B, C, -1)
        v = self.value_conv(x).reshape(B, C, -1).permute(0, 2, 1)

        attn_weights = torch.bmm(q, k) * self.scale
        attn_weights = F.softmax(attn_weights, dim=-1)

        attn_output = torch.bmm(attn_weights, v)
        attn_output = attn_output.permute(0, 2, 1).reshape(B, C, H, W)
        return attn_output
    
class DTCWTTransform(nn.Module):
    def __init__(self, levels=3):
        super(DTCWTTransform, self).__init__()
        self.transform = dtcwt.Transform2d()
        self.levels = levels

    def forward(self, x):
        batch_size, channels, height, width = x.shape
        lowpass_list = []
        highpass_list = [[] for _ in range(self.levels * 4)]  # 4方向性子带每层

        for b in range(batch_size):
            lowpass_channels = []
            for c in range(channels):
                img = x[b, c].cpu().detach().numpy()  # 添加.detach()
                coeffs = self.transform.forward(img, nlevels=self.levels)

                # 保存低频部分
                lowpass_channels.append(torch.tensor(coeffs.lowpass, device=x.device).unsqueeze(0))

                # 保存高频部分
                for l, highpass in enumerate(coeffs.highpasses[:4]):  # 只处理前4个方向性子带
                    highpass_tensor = torch.tensor(highpass, device=x.device)
                    highpass_list[l].append(highpass_tensor)

            # 合并低频部分
            lowpass_list.append(torch.cat(lowpass_channels, dim=0).unsqueeze(0))

        # 合并低频和高频部分
        lowpass = torch.cat(lowpass_list, dim=0)  # (batch_size, channels, low_h, low_w)
        highpasses = [
            torch.stack(level, dim=0) if level else torch.zeros((0, 1, 1), device=x.device) for level in highpass_list
        ]

        return lowpass, highpasses


    def inverse(self, lowpass, highpasses):
        batch_size, channels, low_h, low_w = lowpass.shape
        restored_images = []

        for b in range(batch_size):
            restored_channels = []
            for c in range(channels):
                # 准备单个通道的低频和高频分量
                lowpass_np = lowpass[b, c].cpu().detach().numpy()
                highpasses_np = [highpass[b].cpu().detach().numpy() for highpass in highpasses if highpass.nelement() > 0]

                # 确保我们有正确数量的方向性子带
                if len(highpasses_np) == 4:  # DTCWT通常产生4个方向性子带
                    # 执行逆变换
                    coeffs = dtcwt.Pyramid(lowpass=lowpass_np, highpasses=highpasses_np)
                    restored_img = self.transform.inverse(coeffs)
                    restored_channels.append(torch.tensor(restored_img, device=lowpass.device).unsqueeze(0))
                else:
                    # 如果没有足够的子带，使用零填充
                    zero_highpasses = [np.zeros_like(highpasses_np[0]) for _ in range(4 - len(highpasses_np))]
                    highpasses_np += zero_highpasses
                    coeffs = dtcwt.Pyramid(lowpass=lowpass_np, highpasses=highpasses_np)
                    restored_img = self.transform.inverse(coeffs)
                    restored_channels.append(torch.tensor(restored_img, device=lowpass.device).unsqueeze(0))

            # 合并通道
            restored_image = torch.cat(restored_channels, dim=0).unsqueeze(0)
            restored_images.append(restored_image)

        # 合并所有批次
        restored_images = torch.cat(restored_images, dim=0)
        return restored_images
    
##########################################################################
## Layer Norm
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x = self.dwconv(x)
        x = F.gelu(x)
        x = self.project_out(x)
        return x


##########################################################################
## MDLA improved by designing Multi-scale Convolution
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)

        self.qkv_dwconv_3 = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.qkv_dwconv_5 = nn.Conv2d(dim * 3, dim * 3, kernel_size=5, stride=1, padding=2, groups=dim * 3, bias=bias)
        self.qkv_dwconv_7 = nn.Conv2d(dim * 3, dim * 3, kernel_size=7, stride=1, padding=3, groups=dim * 3, bias=bias)

        self.q_proj = nn.Conv2d(dim * 3, dim, kernel_size=1 ,stride=1, padding=0, bias=bias)
        self.k_proj = nn.Conv2d(dim * 3, dim, kernel_size=1, stride=1, padding=0, bias=bias)
        self.v_proj = nn.Conv2d(dim * 3, dim, kernel_size=1, stride=1, padding=0, bias=bias)

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        x = self.qkv(x)
        qkv_1 = self.qkv_dwconv_3(x)
        q_1, k_1, v_1 = qkv_1.chunk(3, dim=1)

        qkv_2 = self.qkv_dwconv_5(x)
        q_2, k_2, v_2 = qkv_2.chunk(3, dim=1)

        qkv_3 = self.qkv_dwconv_7(x)
        q_3, k_3, v_3 = qkv_3.chunk(3, dim=1)

        q = self.q_proj(torch.cat([q_1, q_2, q_3], dim=1))
        k = self.k_proj(torch.cat([k_1, k_2, k_3], dim=1))
        v = self.v_proj(torch.cat([v_1, v_2, v_3], dim=1))

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)
        return x

class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


##########################################################################
# improved the information Multi-scale conv and lightweights design
class TDN(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 dim=48,
                 num_blocks=[1, 2, 2, 4], #2，3，3，4
                 num_refinement_blocks=4,
                 heads=[1, 2, 4, 8],
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias'
                 ):

        super(TDN, self).__init__()
        self.dtcwt_transform = DTCWTTransform(levels=3)

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.down1_2 = Downsample(dim)
        self.encoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.down2_3 = Downsample(int(dim * 2 ** 1))
        self.encoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.decoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.up3_2 = Upsample(int(dim * 2 ** 2))
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.up2_1 = Upsample(int(dim * 2 ** 1))

        self.decoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.refinement = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])

        self.output = nn.Conv2d(int(dim * 2 ** 1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img):

        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        lowpass1, highpasses1 = self.dtcwt_transform(out_enc_level1)  # DTCWT分解

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)
        lowpass2, highpasses2 = self.dtcwt_transform(out_enc_level2)  # DTCWT分解

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)
        lowpass3, highpasses3 = self.dtcwt_transform(out_enc_level3)  # DTCWT分解

        inp_dec_level3 = out_enc_level3
        out_dec_level3 = self.decoder_level3(inp_dec_level3)
        restored_img3 = self.dtcwt_transform.inverse(lowpass3, highpasses3)  # iDTCWT重建

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)
        restored_img2 = self.dtcwt_transform.inverse(lowpass2, highpasses2)  # iDTCWT重建

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        restored_img1 = self.dtcwt_transform.inverse(lowpass1, highpasses1)  # iDTCWT重建

        out_dec_level1 = self.refinement(out_dec_level1)
        out_dec_level1 = self.output(out_dec_level1)

        return out_dec_level1