import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

###################################################################################################
# version adaptation for PyTorch > 1.7.1
IS_HIGH_VERSION = tuple(map(int, torch.__version__.split('+')[0].split('.'))) > (1, 7, 1)
if IS_HIGH_VERSION:
    import torch.fft
class FocalFrequencyLoss(nn.Module):
    """The torch.nn.Module class that implements focal frequency loss - a
    frequency domain loss function for optimizing generative models.

    Ref:
    Focal Frequency Loss for Image Reconstruction and Synthesis. In ICCV 2021.
    <https://arxiv.org/pdf/2012.12821.pdf>

    Args:
        loss_weight (float): weight for focal frequency loss. Default: 1.0
        alpha (float): the scaling factor alpha of the spectrum weight matrix for flexibility. Default: 1.0
        patch_factor (int): the factor to crop image patches for patch-based focal frequency loss. Default: 1
        ave_spectrum (bool): whether to use minibatch average spectrum. Default: False
        log_matrix (bool): whether to adjust the spectrum weight matrix by logarithm. Default: False
        batch_matrix (bool): whether to calculate the spectrum weight matrix using batch-based statistics. Default: False
    """

    def __init__(self, loss_weight=1.0, alpha=1.0, patch_factor=1, ave_spectrum=False, log_matrix=False, batch_matrix=False):
        super(FocalFrequencyLoss, self).__init__()
        self.loss_weight = loss_weight
        self.alpha = alpha
        self.patch_factor = patch_factor
        self.ave_spectrum = ave_spectrum
        self.log_matrix = log_matrix
        self.batch_matrix = batch_matrix

    def tensor2freq(self, x):
        # crop image patches
        patch_factor = self.patch_factor
        _, _, h, w = x.shape
        assert h % patch_factor == 0 and w % patch_factor == 0, (
            'Patch factor should be divisible by image height and width')
        patch_list = []
        patch_h = h // patch_factor
        patch_w = w // patch_factor
        for i in range(patch_factor):
            for j in range(patch_factor):
                patch_list.append(x[:, :, i * patch_h:(i + 1) * patch_h, j * patch_w:(j + 1) * patch_w])

        # stack to patch tensor
        y = torch.stack(patch_list, 1)

        # perform 2D DFT (real-to-complex, orthonormalization)
        if IS_HIGH_VERSION:
            freq = torch.fft.fft2(y, norm='ortho')
            freq = torch.stack([freq.real, freq.imag], -1)
        else:
            freq = torch.rfft(y, 2, onesided=False, normalized=True)
        return freq

    def loss_formulation(self, recon_freq, real_freq, matrix=None):
        # spectrum weight matrix
        if matrix is not None:
            # if the matrix is predefined
            weight_matrix = matrix.detach()
        else:
            # if the matrix is calculated online: continuous, dynamic, based on current Euclidean distance
            matrix_tmp = (recon_freq - real_freq) ** 2
            matrix_tmp = torch.sqrt(matrix_tmp[..., 0] + matrix_tmp[..., 1]) ** self.alpha

            # whether to adjust the spectrum weight matrix by logarithm
            if self.log_matrix:
                matrix_tmp = torch.log(matrix_tmp + 1.0)

            # whether to calculate the spectrum weight matrix using batch-based statistics
            if self.batch_matrix:
                matrix_tmp = matrix_tmp / matrix_tmp.max()
            else:
                matrix_tmp = matrix_tmp / matrix_tmp.max(-1).values.max(-1).values[:, :, :, None, None]

            matrix_tmp[torch.isnan(matrix_tmp)] = 0.0
            matrix_tmp = torch.clamp(matrix_tmp, min=0.0, max=1.0)
            weight_matrix = matrix_tmp.clone().detach()

        assert weight_matrix.min().item() >= 0 and weight_matrix.max().item() <= 1, (
            'The values of spectrum weight matrix should be in the range [0, 1], '
            'but got Min: %.10f Max: %.10f' % (weight_matrix.min().item(), weight_matrix.max().item()))

        # frequency distance using (squared) Euclidean distance
        tmp = (recon_freq - real_freq) ** 2
        freq_distance = tmp[..., 0] + tmp[..., 1]

        # dynamic spectrum weighting (Hadamard product)
        loss = weight_matrix * freq_distance
        return torch.mean(loss)

    def forward(self, pred, target, matrix=None, **kwargs):
        """Forward function to calculate focal frequency loss.

        Args:
            pred (torch.Tensor): of shape (N, C, H, W). Predicted tensor.
            target (torch.Tensor): of shape (N, C, H, W). Target tensor.
            matrix (torch.Tensor, optional): Element-wise spectrum weight matrix.
                Default: None (If set to None: calculated online, dynamic).
        """
        pred_freq = self.tensor2freq(pred)
        target_freq = self.tensor2freq(target)

        # whether to use minibatch average spectrum
        if self.ave_spectrum:
            pred_freq = torch.mean(pred_freq, 0, keepdim=True)
            target_freq = torch.mean(target_freq, 0, keepdim=True)

        # calculate focal frequency loss
        return self.loss_formulation(pred_freq, target_freq, matrix) * self.loss_weight

##################定义了两个卷积核（也称为滤波器），用于图像处理中的边缘检测############################
Sobel = np.array([[-1, -2, -1],
                  [0, 0, 0],
                  [1, 2, 1]])   #3*3的矩阵，通过计算图像中像素的梯度来检测边缘。Sobel算子分为水平和垂直两个方向，分别对应于矩阵中的行和列。这个矩阵的目的是在图像中寻找变化较大的区域，这些区域通常对应于图像的边缘。
Robert = np.array([[0, 0],
                   [-1, 1]])    #2*2矩阵，通过计算图像中像素的差分来检测边缘
Sobel = torch.Tensor(Sobel)
Robert = torch.Tensor(Robert)

##################计算图像或特征图的梯度############################
def gradient(maps, direction, device='cuda', kernel='sobel'):   #maps=(B,C,H,W);direction = x or y;kernel默认为sobel
    channels = maps.size()[1]       #获取输入特征图的通道数
    if kernel == 'robert':
        smooth_kernel_x = Robert.expand(channels, channels, 2, 2)
        maps = F.pad(maps, (0, 0, 1, 1))        #在图像的右侧和底部各添加一个像素
    elif kernel == 'sobel':
        smooth_kernel_x = Sobel.expand(channels, channels, 3, 3)
        maps = F.pad(maps, (1, 1, 1, 1))        #在图像的左右两侧各添加一个像素
    smooth_kernel_y = smooth_kernel_x.permute(0, 1, 3, 2)       
    if direction == "x":
        kernel = smooth_kernel_x
    elif direction == "y":
        kernel = smooth_kernel_y
    kernel = kernel.to(device=device)
    gradient_orig = torch.abs(F.conv2d(maps, weight=kernel, padding=0))     #使用F.conv2d函数对输入特征图进行卷积操作，得到原始梯度图gradient_orig
    grad_min = torch.min(gradient_orig)
    grad_max = torch.max(gradient_orig)
    grad_norm = torch.div((gradient_orig - grad_min), (grad_max - grad_min + 0.0001))       #计算梯度图的最大值和最小值，然后对梯度图进行归一化处理，使得梯度值在[0,1]范围内，+0.0001防止除0
    return grad_norm        #归一化后的梯度图，形状与输入特征图相同

##################计算图像或特征图的梯度，并对其进行归一化处理#################
def gradient_no_abs(maps, direction, device='cuda', kernel='sobel'):
    channels = maps.size()[1]
    if kernel == 'robert':
        smooth_kernel_x = Robert.expand(channels, channels, 2, 2)
        maps = F.pad(maps, (0, 0, 1, 1))
    elif kernel == 'sobel':
        smooth_kernel_x = Sobel.expand(channels, channels, 3, 3)
        maps = F.pad(maps, (1, 1, 1, 1))
    smooth_kernel_y = smooth_kernel_x.permute(0, 1, 3, 2)
    if direction == "x":
        kernel = smooth_kernel_x
    elif direction == "y":
        kernel = smooth_kernel_y
    kernel = kernel.to(device=device)
    # kernel size is (2, 2) so need pad bottom and right side
    gradient_orig = torch.abs(F.conv2d(maps, weight=kernel, padding=0))     #计算卷积后的绝对值，得到原始梯度图，使得梯度图在0到1之间
    grad_min = torch.min(gradient_orig)
    grad_max = torch.max(gradient_orig)
    grad_norm = torch.div((gradient_orig - grad_min), (grad_max - grad_min + 0.0001))       #归一化后额梯度图，形状与输入特征图相同
    return grad_norm

###############计算图像分解和重建的损失函数###############
class Decom_Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def gradient(self, input_tensor, direction):        #gradient方法用于计算输入张量的梯度，使用两个卷积核分别计算x和y方向上的梯度
        self.smooth_kernel_x = torch.FloatTensor([[0, 0], [-1, 1]]).view((1, 1, 2, 2)).cuda()
        self.smooth_kernel_y = torch.transpose(self.smooth_kernel_x, 2, 3)

        if direction == "x":
            kernel = self.smooth_kernel_x
        elif direction == "y":
            kernel = self.smooth_kernel_y
        grad_out = torch.abs(F.conv2d(input_tensor, kernel, stride=1, padding=1))
        return grad_out

    def ave_gradient(self, input_tensor, direction):        #ave_gradient计算张量的平均梯度，F.avg_pool2d函数对梯度进行平均池化
        return F.avg_pool2d(self.gradient(input_tensor, direction),
                            kernel_size=3, stride=1, padding=1)

    def smooth(self, input_I, input_R):     #smooth方法计算平滑损失，首先将RGB转换为灰度图像，然后计算输入图像和重建图像在x和y方向上的梯度，并使用指数函数对梯度进行加权平均
        input_R = 0.299*input_R[:, 0, :, :] + 0.587*input_R[:, 1, :, :] + 0.114*input_R[:, 2, :, :]
        input_R = torch.unsqueeze(input_R, dim=1)
        return torch.mean(self.gradient(input_I, "x") * torch.exp(-10 * self.ave_gradient(input_R, "x")) +
                          self.gradient(input_I, "y") * torch.exp(-10 * self.ave_gradient(input_R, "y")))

    def forward(self, R_low, R_high, L_low, L_high, I_low, I_high):  

        L_low_3  = torch.cat((L_low, L_low, L_low), dim=1)
        L_high_3 = torch.cat((L_high, L_high, L_high), dim=1)       #将low和high的通道数扩展为3

        ffl = FocalFrequencyLoss(loss_weight=1.0, alpha=1.0)
        self.fre_highloss = ffl(R_high * L_high_3, I_high)  # calculate focal frequency loss
        self.fre_lowloss = ffl(R_high * L_high_3, I_high)  # calculate focal frequency loss

        self.recon_loss_low  = F.l1_loss(R_low * L_low_3,  I_low)
        self.recon_loss_high = F.l1_loss(R_high * L_high_3, I_high)
        self.recon_loss_crs_low  = F.l1_loss(R_high * L_low_3, I_low)
        self.recon_loss_crs_high = F.l1_loss(R_low * L_high_3, I_high)
        self.equal_R_loss = F.l1_loss(R_low,  R_high.detach())

        self.Ismooth_loss_low   = self.smooth(L_low, R_low)
        self.Ismooth_loss_high  = self.smooth(L_high, R_high)       #计算重建损失、交叉重建损失和平滑损失

        self.loss_Decom = self.recon_loss_high + 0.3 * self.recon_loss_low + 0.001 * self.recon_loss_crs_low + \
                          0.001 * self.recon_loss_crs_high + 0.1 * (self.Ismooth_loss_low + self.Ismooth_loss_high) + 0.1 * self.equal_R_loss +\
                          0.2 * (self.fre_highloss +  self.fre_lowloss)

        return self.loss_Decom, self.recon_loss_low + self.recon_loss_high, self.equal_R_loss, self.Ismooth_loss_low + self.Ismooth_loss_high

def normalize_grad(gradient_orig):      #对输入的梯度进行归一化处理（最小-最大归一化方法），提过模型训练的稳定性，加速模型收敛，提高模型泛化能力
    grad_min = torch.min(gradient_orig)
    grad_max = torch.max(gradient_orig)
    grad_norm = torch.div((gradient_orig - grad_min), (grad_max - grad_min + 0.0001))
    return grad_norm