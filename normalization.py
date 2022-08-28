import torch
import torch.nn as nn
from torch.nn import Module, Parameter, init

class tdBatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, alpha=1, affine=True, track_running_stats=True):
        super(tdBatchNorm, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)
        self.alpha = alpha                                # class中新加的一个参数alpha

    def forward(self, input):                             # input(N,C,H,W)
        exponential_average_factor = 0.0
        if self.training and self.track_running_stats:    # training和track_running_stats都为True才更新BN的参数
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1             # 记录完成前向传播的batch数目
                if self.momentum is None:                 # momentum为None，用1/num_batches_tracked代替
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:                                     # use exponential moving average
                    exponential_average_factor = self.momentum

        # calculate running estimates 计算均值和方差的过程
        if self.training:
            mean = input.mean([0, 2, 3])                  #计算均值，出来的维度大小等于channel的数目
            # torch.var 默认算无偏估计，这里先算有偏的，因此需要手动设置unbiased=False
            var = input.var([0, 2, 3], unbiased=False)    # 计算的是有偏方差
            n = input.numel() / input.size(1)             # size(1)是指channel的数目  n=N*H*W
            with torch.no_grad():                         # 计算均值和方差的过程不需要梯度传输
                self.running_mean = exponential_average_factor * mean \
                                    + (1 - exponential_average_factor) * self.running_mean
                # update running_var with unbiased var 这里通过有偏方差和无偏方差的关系，又转换成了无偏方差
                self.running_var = exponential_average_factor * var * n / (n - 1) \
                                   + (1 - exponential_average_factor) * self.running_var
        else:                                             # 不处于训练模式就固定running_mean和running_var的值
            mean = self.running_mean
            var = self.running_var


        input = self.alpha * (input - mean[None, :, None, None])/(torch.sqrt(var[None, :, None, None] + self.eps)) # 用None扩充维度，然后与原输入tensor做相应运算实现规范化
        if self.affine:
            input = input * self.weight[None, :, None, None] + self.bias[None, :, None, None]

        return input

class tdInstanceNorm(nn.InstanceNorm2d):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, alpha=1, affine=True, track_running_stats=True):
        super(tdInstanceNorm, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)
        self.alpha = alpha                                # class中新加的一个参数alpha

    def forward(self, input):                             # input(N,C,H,W)
        exponential_average_factor = 0.0
        if self.training and self.track_running_stats:    # training和track_running_stats都为True才更新BN的参数
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1             # 记录完成前向传播的batch数目
                if self.momentum is None:                 # momentum为None，用1/num_batches_tracked代替
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:                                     # use exponential moving average
                    exponential_average_factor = self.momentum

        # calculate running estimates 计算均值和方差的过程
        if self.training:
            mean = input.mean([2, 3])                  #计算均值，出来的维度大小等于channel的数目
            # torch.var 默认算无偏估计，这里先算有偏的，因此需要手动设置unbiased=False
            var = input.var([2, 3], unbiased=False)    # 计算的是有偏方差
            n = input.numel() / input.size(1)/  input.size(0)           # size(1)是指channel的数目  n=N*H*W
            with torch.no_grad():                         # 计算均值和方差的过程不需要梯度传输
                self.running_mean = exponential_average_factor * mean \
                                    + (1 - exponential_average_factor) * self.running_mean
                # update running_var with unbiased var 这里通过有偏方差和无偏方差的关系，又转换成了无偏方差
                self.running_var = exponential_average_factor * var * n / (n - 1) \
                                   + (1 - exponential_average_factor) * self.running_var
        else:                                             # 不处于训练模式就固定running_mean和running_var的值
            mean = self.running_mean
            var = self.running_var


        input = self.alpha * (input - mean[:, :, None, None])/(torch.sqrt(var[:, :, None, None] + self.eps)) # 用None扩充维度，然后与原输入tensor做相应运算实现规范化
        if self.affine:
            input = input * self.weight[:, :, None, None] + self.bias[:, :, None, None]

        return input


class _ComplexInstanceNorm(nn.Module):

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(_ComplexInstanceNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.weight = Parameter(torch.Tensor(num_features, 3))
            self.bias = Parameter(torch.Tensor(num_features, 2))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features, dtype=torch.complex64))
            self.register_buffer('running_covar', torch.zeros(num_features,3))
            self.running_covar[:, 0] = 1.4142135623730951
            self.running_covar[:, 1] = 1.4142135623730951
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_covar', None)
            self.register_parameter('num_batches_tracked', None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_covar.zero_()
            self.running_covar[:, 0] = 1.4142135623730951
            self.running_covar[:, 1] = 1.4142135623730951
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            init.constant_(self.weight[:, :2], 1.4142135623730951)
            init.zeros_(self.weight[:, 2])
            init.zeros_(self.bias)


class ComplexInstanceNorm2d(_ComplexInstanceNorm):

    def forward(self, inputr, inputi):
        input = torch.complex(inputr, inputi)
        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        if self.training or (not self.training and not self.track_running_stats):
            # calculate mean of real and imaginary part
            # mean does not support automatic differentiation for outputs with complex dtype.
            mean_r = input.real.mean([2, 3])
            mean_i = input.imag.mean([2, 3])
            mean = mean_r + 1j * mean_i
        else:
            mean = self.running_mean

        if self.training and self.track_running_stats:
            # update running mean
            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean \
                                    + (1 - exponential_average_factor) * self.running_mean

        input = input - mean[:, :, None, None]

        if self.training or (not self.training and not self.track_running_stats):
            # Elements of the covariance matrix (biased for train)
            n = input.numel() / input.size(1)/input.size(0)
            Crr = 1. / n * input.real.pow(2).sum(dim=[2, 3]) + self.eps
            Cii = 1. / n * input.imag.pow(2).sum(dim=[2, 3]) + self.eps
            Cri = (input.real.mul(input.imag)).mean(dim=[2, 3])
        else:
            Crr = self.running_covar[:, 0] + self.eps
            Cii = self.running_covar[:, 1] + self.eps
            Cri = self.running_covar[:, 2]  # +self.eps

        if self.training and self.track_running_stats:
            with torch.no_grad():
                if self.running_covar.ndim==2:
                    expand_dim = self.running_mean.size(0)
                    self.running_covar = self.running_covar.repeat(expand_dim,1,1)
                    self.running_covar[:, :, 0] = exponential_average_factor * Crr * n / (n - 1) \
                                                  + (1 - exponential_average_factor) * self.running_covar[:, :, 0]

                    self.running_covar[:, :, 1] = exponential_average_factor * Cii * n / (n - 1) \
                                                  + (1 - exponential_average_factor) * self.running_covar[:, :, 1]

                    self.running_covar[:, :, 2] = exponential_average_factor * Cri * n / (n - 1) \
                                                  + (1 - exponential_average_factor) * self.running_covar[:, :, 2]
                elif self.running_covar.ndim==3:

                    self.running_covar[:,:, 0] = exponential_average_factor * Crr * n / (n - 1) \
                                               + (1 - exponential_average_factor) * self.running_covar[:,:, 0]

                    self.running_covar[:,:, 1] = exponential_average_factor * Cii * n / (n - 1) \
                                               + (1 - exponential_average_factor) * self.running_covar[:,:, 1]

                    self.running_covar[:,:, 2] = exponential_average_factor * Cri * n / (n - 1) \
                                               + (1 - exponential_average_factor) * self.running_covar[:,:, 2]

        # calculate the inverse square root the covariance matrix
        det = Crr * Cii - Cri.pow(2)
        s = torch.sqrt(det)
        t = torch.sqrt(Cii + Crr + 2 * s)
        inverse_st = 1.0 / (s * t)
        Rrr = (Cii + s) * inverse_st
        Rii = (Crr + s) * inverse_st
        Rri = -Cri * inverse_st

        inputr = Rrr[:,:, None, None] * input.real + Rri[:, :,None, None] * input.imag
        inputi = Rii[:, :, None, None] * input.imag + Rri[:, :, None, None] * input.real

        if self.affine:
            # input = (self.weight[None,:,0,None,None]*input.real+self.weight[None,:,2,None,None]*input.imag+\
            #         self.bias[None,:,0,None,None]).type(torch.complex64) \
            #         +1j*(self.weight[None,:,2,None,None]*input.real+self.weight[None,:,1,None,None]*input.imag+\
            #         self.bias[None,:,1,None,None]).type(torch.complex64)

            inputr = self.weight[None, :, 0, None, None] * input.real + self.weight[None, :, 2, None,
                                                                        None] * input.imag + \
                     self.bias[None, :, 0, None, None]
            inputi = self.weight[None, :, 2, None, None] * input.real + self.weight[None, :, 1, None,
                                                                        None] * input.imag + \
                     self.bias[None, :, 1, None, None]
        # del Crr, Cri, Cii, Rrr, Rii, Rri, det, s, t
        return inputr, inputi

if __name__ == "__main__":

    x = torch.randn((32,3,224,224)) # [B,C,H,W]
    xi = torch.randn((32,3,224,224))
    b, c, h, w = x.shape

    # pytorch
    bn = nn.BatchNorm2d(c, eps=1e-10, affine=True, track_running_stats=True)
    bn2 = tdBatchNorm(c, eps=1e-10, affine=True, track_running_stats=True)
    # affine=False 关闭线性映射
    # track_running_stats=False 只计算当前的平均值与方差，而非更新全局统计量


    ##  调用ComplexInstanceNorm2d时，注意数据的格式是[B,C,H,W]
    complex_IN=ComplexInstanceNorm2d(c, eps=1e-12, affine=False, track_running_stats=True)
    complex_yin = complex_IN(x,xi)
    In = nn.InstanceNorm2d(c, eps=1e-12, affine=False, track_running_stats=True)
    In2 = tdInstanceNorm(c, eps=1e-12, affine=True, track_running_stats=True)
    y_in = In(x)
    y_in2 = In2(x)
    # y = bn(x)
    # y2 = bn2(x)

    # 为了方便理解，这里使用了einops库实现
    # x_ = rearrange(x, 'b c h w -> (b w h) c')
    # mean = rearrange(x_.mean(dim=0), 'c -> 1 c 1 1')
    # std = rearrange(x_.std(dim=0), 'c -> 1 c 1 1')

    # y_ = (x-mean)/std

    # 输出差别
    print('diff={}'.format(torch.abs(y_in-y_in2).max()))
    # diff=1.9073486328125e-06