import torch
import torch.nn as nn
import torch.nn.functional as F
from model import source_code as S

'''ESA块的编写'''
class ESA(nn.Module):#ESA的相关内容在RFAnet中
    #ESA:卷积1降通道维数，卷积2降空间尺寸，随后池化，随后3个3*3卷积形成conv-groups，随后上采样，conv4融合特征升维。sig激活。
    def __init__(self, n_feats, conv):
        super(ESA, self).__init__()
        f = n_feats // 4
        self.conv1 = conv(n_feats, f, kernel_size=1)#对输入使用1x1卷积降通道数,为输入通道1/4
        self.conv_f = conv(f, f, kernel_size=1)#con4融合c3和cf特征，cf需要conv_f使c1_与c3同通道和尺寸
        self.conv2 = conv(f, f, kernel_size=3, stride=2, padding=0)#使用步长为2的卷积，降特征空间尺寸
        self.conv_max = conv(f, f, kernel_size=3, padding=1)#conv_max与conv3和conv3_都是conv groups组成部分，卷积一样
        self.conv3 = conv(f, f, kernel_size=3, padding=1)
        self.conv3_ = conv(f, f, kernel_size=3, padding=1)
        self.conv4 = conv(f, n_feats, kernel_size=1)#特征融合，扩增通道
        self.sigmoid = nn.Sigmoid()#激活
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        c1_ = self.conv1(x)
        c1 = self.conv2(c1_)
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)
        v_range = self.relu(self.conv_max(v_max))
        c3 = self.relu(self.conv3(v_range))
        c3 = self.conv3_(c3)
        c3 = F.interpolate(c3, (x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3 + cf)
        m = self.sigmoid(c4)
        return x * m

'''RFDB块，此处，3*3卷积未换成SRB'''
class RFDB(nn.Module):
    #定义RFDB模块，但此处编写与论文不符，直接使用了3*3卷积，残差未加，后期视网络修改再更改。
    def __init__(self,in_channels,distillation_rate=0.25):
        super(RFDB,self).__init__()
        self.dc = self.distilled_channels = in_channels // 2  #蒸馏后的通道数
        self.rc = self.remaining_channels = in_channels       #送入下一层蒸馏的通道数
        self.c1_d=S.conv_layer(in_channels,self.dc,1)           #padding=0
        self.c1_r=S.conv_layer(in_channels,self.rc,3)           #padding=1
        self.c2_d=S.conv_layer(self.remaining_channels,self.dc,1)
        self.c2_r=S.conv_layer(self.remaining_channels,self.rc,3)
        self.c3_d=S.conv_layer(self.remaining_channels,self.dc,1)
        self.c3_r=S.conv_layer(self.remaining_channels,self.rc,3)
        self.c4=S.conv_layer(self.remaining_channels,self.dc,3)
        self.act=S.activation('lrelu')
        self.c5=S.conv_layer(self.dc*4,in_channels,1)
        self.esa = ESA(in_channels, nn.Conv2d)

    def forward(self, input):
        distilled_c1 = self.act(self.c1_d(input))
        r_c1 = (self.c1_r(input))
        r_c1 = self.act(r_c1+input)

        distilled_c2 = self.act(self.c2_d(r_c1))
        r_c2 = (self.c2_r(r_c1))
        r_c2 = self.act(r_c2+r_c1)

        distilled_c3 = self.act(self.c3_d(r_c2))
        r_c3 = (self.c3_r(r_c2))
        r_c3 = self.act(r_c3+r_c2)

        r_c4 = self.act(self.c4(r_c3))

        out = torch.cat([distilled_c1, distilled_c2, distilled_c3, r_c4], dim=1)
        out_fused = self.esa(self.c5(out))

        return out_fused

'''亚像素卷积块'''
def pixelshuffle_block(in_channels, out_channels, upscale_factor=2, kernel_size=3, stride=1):
    # 首先通过卷积将通道数扩展为 scaling factor^2 倍
    conv = S.conv_layer(in_channels, out_channels * (upscale_factor ** 2), kernel_size, stride)
    # 进行像素清洗，合并相关通道数据
    pixel_shuffle = nn.PixelShuffle(upscale_factor)
    return S.sequential(conv, pixel_shuffle, nn.PReLU)

'''各模块构建完毕，开始构建网络'''
class RFDN(nn.Module):
    def __init__(self,in_channels=3,nf=52,num_modules=6,out_channels=3):
        super(RFDN, self).__init__()
        #创建浅层特征提取层
        self.fec=S.conv_layer(in_channels,nf, kernel_size=3)
        #堆叠RFDB块
        self.RFDB1=RFDB(in_channels=nf)
        self.RFDB2=RFDB(in_channels=nf)
        self.RFDB3=RFDB(in_channels=nf)
        self.RFDB4=RFDB(in_channels=nf)
        self.RFDB5=RFDB(in_channels=nf)
        self.RFDB6=RFDB(in_channels=nf)
        self.c = S.conv_block(nf * num_modules, nf, kernel_size=1, act_type='lrelu')#特征聚合
        self.LR_conv = S.conv_layer(nf, nf, kernel_size=3)
        self.upsampler = pixelshuffle_block(nf,out_channels,upscale_factor=2)

    def forward(self, input):
        out_fea = self.fec(input)
        out_B1 = self.RFDB1(out_fea)
        out_B2 = self.RFDB2(out_B1)
        out_B3 = self.RFDB3(out_B2)
        out_B4 = self.RFDB4(out_B3)
        out_B5=self.RFDB5(out_B4)
        out_B6=self.RFDB6(out_B5)
        out_B = self.c(torch.cat([out_B1, out_B2, out_B3, out_B4,out_B5,out_B6], dim=1))#特征聚合
        out_lr = self.LR_conv(out_B) + out_fea

        output = self.upsampler(out_lr)

        return output






