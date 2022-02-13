import torch
import torch.nn as nn
from model import block as B
import torch.nn.functional as F

'''亚像素卷积块'''
def pixelshuffle_block(in_channels, out_channels, upscale_factor, kernel_size=3, stride=1):
    # 首先通过卷积将通道数扩展为 scaling factor^2 倍
    conv = B.conv_layer(in_channels, out_channels * (upscale_factor ** 2), kernel_size, stride)
    # 进行像素清洗，合并相关通道数据
    pixel_shuffle = nn.PixelShuffle(upscale_factor)
    return B.sequential(conv, pixel_shuffle, nn.PReLU)


'''空间、通道特征交叉融合模块'''
class FIFM(nn.Module):#cross-fussion moudle
    def __init__(self, in_channels, out_channels, scale=2):
        super(FIFM, self).__init__()
        self.CFE = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=int(scale),
                             stride=int(scale), padding=0, bias=False)
        self.SFE = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1,
                             dilation=int(scale), padding=int(scale), bias=False)
        self.STC = nn.Conv2d(in_channels, in_channels, kernel_size=int(scale), stride=int(scale), padding=0,
                             bias=False)
        #self.CTS = pixelshuffle_block(in_channels=in_channels, out_channels=in_channels, upscale_factor=scale)
        self.chaconv = nn.Conv2d(2 * in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=False)
        #self.spaconv = nn.Conv2d(2 * in_channels, in_channels, kernel_size=3, stride=1, dilation=int(scale),padding=int(scale), bias=False)
        self.conv1=B.conv_layer(in_channels,out_channels,kernel_size=1)
        #self.conv2=B.conv_layer(in_channels,out_channels,kernel_size=1)
        self.ReLU = B.activation('lrelu')

    def forward(self, x):
        out_CFE = self.CFE(x)
        out_SFE = self.SFE(x)
        out_CFE2 = self.STC(out_SFE)
        #out_SFE2 = self.CTS(out_CFE)
        feature_c = torch.cat((out_CFE, out_CFE2), 1)
        #feature_s = torch.cat((out_SFE, out_SFE2), 1)
        out_c = self.ReLU(self.chaconv(feature_c)) + out_CFE
        #out_s = self.ReLU(self.spaconv(feature_s)) + out_SFE
        out_c=self.conv1(out_c)
        #out_s=self.conv2(out_s)
        return out_c


class CAM(nn.Module):
    def __init__(self, in_planes):
        super(CAM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        m=self.sigmoid(out)
        return x*m

class SAM(nn.Module):
    def __init__(self, kernel_size=7):
        super(SAM, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        m = torch.cat([avg_out, max_out], dim=1)
        m = self.conv1(m)
        m=self.sigmoid(m)
        return x*m


class ESA(nn.Module):#spation attention
    def __init__(self, n_feats, conv):
        super(ESA, self).__init__()
        f = n_feats // 4
        self.conv1 = conv(n_feats, f, kernel_size=1)
        self.conv_f = conv(f, f, kernel_size=1)
        self.conv_max = conv(f, f, kernel_size=3, padding=1)
        self.conv2 = conv(f, f, kernel_size=3, stride=2, padding=0)
        self.conv3 = conv(f, f, kernel_size=3, padding=1)
        self.conv3_ = conv(f, f, kernel_size=3, padding=1)
        self.conv4 = conv(f, n_feats, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        c1_ = (self.conv1(x))
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

class CDM(nn.Module):
    def __init__(self, in_channels):
        super(CDM, self).__init__()
        self.dc = self.distilled_channels = in_channels // 2
        self.rc = self.remaining_channels = in_channels
        self.act = B.activation('lrelu')

        #out_c 部分
        self.c1_d=B.conv_layer(in_channels,self.dc,1)
        self.SAM1 = SAM()
        self.c1_r=B.conv_layer(in_channels,self.rc,3)
        self.SAM1_= SAM()

        self.c2_d = B.conv_layer(self.rc, self.dc, 1)
        self.SAM2 = SAM()
        self.c2_r = B.conv_layer(self.rc, self.rc, 3)
        self.SAM2_= SAM()

        self.c3_d = B.conv_layer(self.rc, self.dc, 1)
        self.SAM3 = SAM()
        self.c3_r = B.conv_layer(self.rc, self.rc, 3)
        self.SAM3_=SAM()

        self.c4_d = B.conv_layer(self.rc, self.dc, 1)
        self.SAM4 = SAM()

        self.c5 = B.conv_layer(self.dc * 4, in_channels, 1)
        self.esa = ESA(in_channels, nn.Conv2d)

    def forward(self, input):
        distilled_c1 = self.act(self.SAM1(self.c1_d(input)))
        r_c1 =self.SAM1_(self.c1_r(input))
        r_c1 =self.act(r_c1+input)

        distilled_c2 = self.act(self.SAM2(self.c2_d(r_c1)))
        r_c2 = self.SAM2_(self.c2_r(r_c1))
        r_c2 = self.act(r_c2 + r_c1)

        distilled_c3 = self.act(self.SAM3(self.c3_d(r_c2)))
        r_c3 = self.SAM3_(self.c3_r(r_c2))
        r_c3 = self.act(r_c3 + r_c2)

        distilled_c4 = self.act(self.SAM4(self.c4_d(r_c3)))
        out = torch.cat([distilled_c1, distilled_c2, distilled_c3,distilled_c4], dim=1)
        out_fused = self.esa(self.c5(out))
        return out_fused

class MDASR(nn.Module):
    def __init__(self, in_channels=3 ,out_channels=52,num_modules=6,scale=2):
        super(MDASR, self).__init__()
        self.pixelshuffle = pixelshuffle_block(in_channels, in_channels, upscale_factor=scale)
        self.FIFM=FIFM(in_channels=in_channels,out_channels=out_channels)
        self.CDM1=CDM(out_channels)
        self.CDM2=CDM(out_channels)
        self.CDM3=CDM(out_channels)
        self.CDM4=CDM(out_channels)
        self.CDM5=CDM(out_channels)
        self.CDM6=CDM(out_channels)
        self.c = B.conv_block(out_channels * num_modules,out_channels, kernel_size=1, act_type='lrelu')  # 特征聚合
        self.LR_conv = B.conv_layer(out_channels, out_channels, kernel_size=3)
        self.upsampler = pixelshuffle_block(out_channels, in_channels, upscale_factor=2)
        self.conv=B.conv_layer(in_channels, in_channels, kernel_size=1)

    def forward(self,input):
        out_pixel=self.pixelshuffle(input)
        out_FIFM=self.FIFM(out_pixel)
        out_B1 = self.CDM1(out_FIFM)
        out_B2 = self.CDM2(out_B1)
        out_B3 = self.CDM3(out_B2)
        out_B4 = self.CDM4(out_B3)
        out_B5 = self.CDM5(out_B4)
        out_B6 = self.CDM6(out_B5)
        out_B = self.c(torch.cat([out_B1, out_B2, out_B3, out_B4, out_B5, out_B6], dim=1))  # 特征聚合
        out_lr = self.LR_conv(out_B) + out_FIFM
        output = self.upsampler(out_lr)
        output = self.conv(output+out_pixel)
        return output



