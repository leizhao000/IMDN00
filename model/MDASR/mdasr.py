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
    def __init__(self,in_channels,out_channels,scale=2):
        super(FIFM, self).__init__()
        self.pixelshuffle=pixelshuffle_block(in_channels, in_channels,upscale_factor=scale)
        self.CFE=nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=int(scale), stride=int(scale), padding=0, bias=False)
        self.SFE=nn.Conv2d(in_channels=in_channels, out_channels=in_channels,kernel_size=3,stride=1,dilation=int(scale), padding=int(scale), bias=False)
        self.STC=nn.Conv2d(in_channels, in_channels, kernel_size=int(scale), stride=int(scale), padding=0, bias=False)
        self.chaconv= nn.Conv2d(2 * in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv=B.conv_layer(in_channels,out_channels,kernel_size=1)
        self.ReLU = nn.ReLU(inplace=True)

    def forward(self,x):
        out_FFM=self.pixelshuffle(x)
        out_CFE=self.CFE(out_FFM)
        out_SFE=self.SFE(out_FFM)
        out_CFE2=self.STC(out_SFE)
        feature_c=torch.cat((out_CFE,out_CFE2),1)
        out_c=self.ReLU(self.chaconv(feature_c))+out_CFE
        out_c=self.conv(out_c)
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
        self.conv = B.conv_layer(in_planes, in_planes, 1)
        self.act = B.activation('lrelu')

    def forward(self, x):
        out=self.conv(x)
        avg_out = self.fc(self.avg_pool(out))
        max_out = self.fc(self.max_pool(out))
        out = avg_out + max_out
        m=self.sigmoid(out)
        out=out*m
        out=x+out
        out=self.act(out)
        return out

class SAM(nn.Module):#spation attention
    def __init__(self, n_feats, conv):
        super(SAM, self).__init__()
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


class MFDB(nn.Module):
    def __init__(self,in_channels):
        super(MFDB,self).__init__()
        self.dc= int(in_channels // 4)
        self.rc=int(in_channels-self.dc)

        self.c1=B.conv_layer(in_channels, in_channels, 3)
        self.c2=B.conv_layer(self.rc, self.dc, 1)
        self.c3=B.conv_layer(self.rc, self.dc, 1)
        self.c4=B.conv_layer(self.rc, self.dc, 1)

        self.SAM1 = SAM(n_feats=self.dc, conv=nn.Conv2d)
        self.SAM2 = SAM(n_feats=self.dc, conv=nn.Conv2d)
        self.SAM3 = SAM(n_feats=self.dc, conv=nn.Conv2d)
        self.SAM4 = SAM(n_feats=self.dc, conv=nn.Conv2d)

        self.CAM1 = CAM(in_planes=in_channels)
        self.CAM2 = CAM(in_planes=in_channels)
        self.CAM3 = CAM(in_planes=in_channels)
        self.CAM4 = CAM(in_planes=in_channels)


        self.conv = B.conv_layer(self.dc * 4, in_channels, 1)
        self.SAM5=SAM(n_feats=in_channels,conv=nn.Conv2d)
        self.act = B.activation('lrelu')

    def forward(self, out_c):
        out_d1=self.SAM1(self.c1_d(out_c))
        out_d1=self.act(out_d1)
        out_r1=self.CAM1(out_c)

        out_d2=self.SAM2(self.c2_d(out_r1))
        out_d2 = self.act(out_d2)
        out_r2=self.CAM2(out_r1)

        out_d3=self.SAM3(self.c3_d(out_r2))
        out_d3 = self.act(out_d3)
        out_r3=self.CAM3(out_r2)

        out_d4=self.SAM4(self.c4_d(out_r3))
        out_d4 = self.act(out_d4)

        out_fused = torch.cat([out_d1, out_d2, out_d3, out_d4], dim=1)
        out_fused = self.conv(out_fused)
        out_fused=self.SAM5(out_fused)

        out=out_fused+out_c
        out=self.act(out)
        return  out

class MDASR(nn.Module):
    def __init__(self, in_channels,out_channels,num_modules=6):
        super(MDASR, self).__init__()
        self.FIFM=FIFM(in_channels=in_channels,out_channels=out_channels)
        self.B1 = MFDB(in_channels=out_channels)
        self.B2 = MFDB(in_channels=out_channels)
        self.B3 = MFDB(in_channels=out_channels)
        self.B4 = MFDB(in_channels=out_channels)
        self.B5 = MFDB(in_channels=out_channels)
        self.B6 = MFDB(in_channels=out_channels)
        self.c1 = B.conv_block(out_channels * num_modules, out_channels, kernel_size=1, act_type='lrelu')
        self.LR_conv = B.conv_layer(out_channels, out_channels, kernel_size=3)
        self.pixelshuffle = pixelshuffle_block(out_channels, in_channels, upscale_factor=2)

    def forward(self,input):
        out_c = self.FIFM(input)
        out_B1=self.B1(out_c)
        out_B2=self.B2(out_B1)
        out_B3=self.B3(out_B2)
        out_B4=self.B4(out_B3)
        out_B5=self.B5(out_B4)
        out_B6 = self.B6(out_B5)
        out_B= self.c1(torch.cat([out_B1, out_B2,out_B3, out_B4,out_B5,out_B6], dim=1))
        out_lr=self.LR_conv(out_B)+ out_c
        out_sr=self.pixelshuffle(out_lr)
        return out_sr

