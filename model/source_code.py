import torch.nn as nn
from collections import  OrderedDict

'''卷积层定义,补0根据卷积大小和dilation确定'''
def conv_layer(in_channels, out_channels, kernel_size, dilation=1):
    padding= int((kernel_size - 1) // 2) * dilation
    return nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,padding=padding)

'''Conv block块,几个卷积的聚合'''
def conv_block(in_nc, out_nc, kernel_size, dilation=1, pad_type='zero', norm_type=None, act_type='relu'):
    padding = get_valid_padding(kernel_size, dilation)
    p =pad(pad_type, padding) if pad_type != 'zero' else None
    padding = padding if pad_type == 'zero' else 0

    c = nn.Conv2d(in_nc, out_nc, kernel_size=kernel_size, padding=padding,dilation=dilation)
    a = activation(act_type) if act_type else None
    n = norm(norm_type, out_nc) if norm_type else None
    return sequential(p, c, n, a)

'''激活层选择，在relu，lrelu，prelu，优先选择relu，当ReLU效果不太理想，下一个建议是试试LeakyReLU，当且仅当数据量庞大时使用prelu。'''
def activation(act_type,inplace=True,neg_slope=0.05, n_prelu=1):
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace=inplace)
    elif act_type == 'lrelu':
        layer = nn.LeakyReLU(negative_slope=neg_slope, inplace=inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
        #num_parameters：要学习a的数量，可以输入两种值，1或者输入的通道数，默认是1  init:a的初始值，默认0.25
    else:
        raise NotImplementedError('activation layer [{:s}] is not found'.format(act_type))
    return layer

'''sequential函数封装了nn.Sequential，这是源码，看不懂就略过'''
def sequential(*args):
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0] # No sequential is needed.
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)

'''定义了两个函数，针对不同模式下的padding'''
def pad(pad_type, padding):#padding填充方式
    pad_type = pad_type.lower()
    if padding == 0:
        return None
    if pad_type == 'reflect':#反射填充
        layer = nn.ReflectionPad2d(padding)
    elif pad_type == 'replicate':#重复填充
        layer = nn.ReplicationPad2d(padding)
    else:
        raise NotImplementedError('padding layer [{:s}] is not implemented'.format(pad_type))
        #它的意思是如果这个方法没有被子类重写，但是调用了，就会报错。
    return layer

def get_valid_padding(kernel_size, dilation):#padding大小设置
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = (kernel_size - 1) // 2
    return padding

'''定义了函数，针对不同模式下的norm'''
def norm(norm_type, nc):
    norm_type = norm_type.lower()
    if norm_type == 'batch':
        layer = nn.BatchNorm2d(nc, affine=True)#nc为输入的数据的通道数
        #BN归一化算NHW的均值，对小batchsize效果不好。此函数的作用是对输入的每个batch数据做归一化处理,目的是数据合理分布，加速计算过程
    elif norm_type == 'instance':
        layer = nn.InstanceNorm2d(nc, affine=False)
        #一个channel内做归一化，算H*W的均值。用在风格化迁移，可以加速模型收敛，并且保持每个图像实例之间的独立。
    else:
        raise NotImplementedError('normalization layer [{:s}] is not found'.format(norm_type))
    return layer

class ShortcutBlock(nn.Module):
    def __init__(self, submodule):
        super(ShortcutBlock, self).__init__()
        self.sub = submodule

    def forward(self, x):
        output = x + self.sub(x)
        return output

def mean_channels(F):
    assert(F.dim() == 4)
    spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (F.size(2) * F.size(3))

def stdv_channels(F):
    assert(F.dim() == 4)
    F_mean = mean_channels(F)
    F_variance = (F - F_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (F.size(2) * F.size(3))
    return F_variance.pow(0.5)



