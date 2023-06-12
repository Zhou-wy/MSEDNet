"""
description:
version:
Author: zwy
Date: 2023-05-05 15:41:37
LastEditors: zwy
LastEditTime: 2023-06-05 16:25:24
"""

import os
import torch
import torch.nn as nn
from resnet import Backbone_ResNet152_in3, Backbone_ResNet50_in3, Backbone_ResNet34_in3
import torch.nn.functional as F
import numpy as np


class CAM_Module(nn.Module):
    """ Channel attention module"""

    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        # self.relu = nn.ReLU(inplace=True)
        self.relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=4):
        super(ChannelAttention, self).__init__()
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = max_out
        x = self.conv1(x)
        return self.sigmoid(x)


class CorrelationModule(nn.Module):
    def __init__(self, all_channel=64):
        super(CorrelationModule, self).__init__()
        self.linear_e = nn.Linear(all_channel, all_channel, bias=False)
        self.channel = all_channel
        self.fusion = BasicConv2d(all_channel, all_channel, kernel_size=3, padding=1)

    def forward(self, exemplar, query):  # exemplar: middle, query: rgb or T
        fea_size = exemplar.size()[2:]
        all_dim = fea_size[0] * fea_size[1]
        exemplar_flat = exemplar.view(-1, self.channel, all_dim)  # N,C,H*W
        query_flat = query.view(-1, self.channel, all_dim)
        exemplar_t = torch.transpose(exemplar_flat, 1, 2).contiguous()  # batchsize x dim x num, N,H*W,C
        exemplar_corr = self.linear_e(exemplar_t)  #
        A = torch.bmm(exemplar_corr, query_flat)
        B = F.softmax(torch.transpose(A, 1, 2), dim=1)
        exemplar_att = torch.bmm(query_flat, B).contiguous()

        exemplar_att = exemplar_att.view(-1, self.channel, fea_size[0], fea_size[1])
        exemplar_out = self.fusion(exemplar_att)

        return exemplar_out


# Convolution operation
class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, is_last=False):
        super(ConvLayer, self).__init__()
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.dropout = nn.Dropout2d(p=0.5)
        self.is_last = is_last

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        if self.is_last is False:
            # out = F.normalize(out)
            out = F.relu(out, inplace=True)
            # out = self.dropout(out)
        return out


class CRF(torch.nn.Module):
    def __init__(self, channels):
        super(CRF, self).__init__()
        ws = [3, 3, 3, 3]
        self.conv_fusion = ConvLayer(2 * channels, channels, 3, 1)

        self.conv_sum = ConvLayer(channels, channels, 3, 1)
        self.conv_mul = ConvLayer(channels, channels, 3, 1)

        block = []
        block += [ConvLayer(2 * channels, channels, 1, 1),
                  ConvLayer(channels, channels, 3, 1),
                  ConvLayer(channels, channels, 3, 1)]
        self.bottelblock = nn.Sequential(*block)

    def forward(self, x_ir, x_vi):
        # initial fusion - conv
        # print('conv')
        f_sum = x_ir + x_vi
        f_mul = x_ir * x_vi
        f_init = torch.cat([f_sum, f_sum], 1)
        f_init = self.conv_fusion(f_init)

        out_ir = self.conv_sum(f_sum)
        out_vi = self.conv_mul(f_mul)
        out = torch.cat([out_ir, out_vi], 1)
        out = self.bottelblock(out)
        out = f_init + out
        return out


class CLM(nn.Module):
    def __init__(self, all_channel=64):
        super(CLM, self).__init__()
        self.corr_x_2_x_ir = CorrelationModule(all_channel)
        self.corr_ir_2_x_ir = CorrelationModule(all_channel)
        self.smooth1 = BasicConv2d(all_channel, all_channel, kernel_size=3, padding=1)
        self.smooth2 = BasicConv2d(all_channel, all_channel, kernel_size=3, padding=1)
        self.fusion = BasicConv2d(2 * all_channel, all_channel, kernel_size=3, padding=1)
        self.pred = nn.Conv2d(all_channel, 2, kernel_size=3, padding=1, bias=True)

    def forward(self, x, x_ir, ir):  # exemplar: middle, query: rgb or T
        corr_x_2_x_ir = self.corr_x_2_x_ir(x_ir, x)
        corr_ir_2_x_ir = self.corr_ir_2_x_ir(x_ir, ir)

        summation = self.smooth1(corr_x_2_x_ir + corr_ir_2_x_ir)
        multiplication = self.smooth2(corr_x_2_x_ir * corr_ir_2_x_ir)

        fusion = self.fusion(torch.cat([summation, multiplication], 1))
        sal_pred = self.pred(fusion)

        return fusion, sal_pred


class SCA(nn.Module):
    def __init__(self, all_channel=64):
        super(SCA, self).__init__()
        # self.conv1 = BasicConv2d(all_channel, all_channel, kernel_size=3, padding=1)
        self.conv2 = BasicConv2d(all_channel, all_channel, kernel_size=3, padding=1)
        self.sa = SpatialAttention()
        # self-channel attention
        self.cam = CAM_Module(all_channel)

    def forward(self, x, ir):
        multiplication = x * ir
        summation = self.conv2(x + ir)

        sa = self.sa(multiplication)
        summation_sa = summation.mul(sa)

        sc_feat = self.cam(summation_sa)

        return sc_feat


class EDS(nn.Module):
    def __init__(self, all_channel=64):
        super(EDS, self).__init__()
        self.conv1 = BasicConv2d(all_channel, all_channel, kernel_size=3, padding=1)
        self.conv2 = BasicConv2d(all_channel, all_channel, kernel_size=3, padding=1)
        self.dconv1 = BasicConv2d(all_channel, int(all_channel / 4), kernel_size=3, padding=1)
        self.dconv2 = BasicConv2d(all_channel, int(all_channel / 4), kernel_size=3, dilation=3, padding=3)
        self.dconv3 = BasicConv2d(all_channel, int(all_channel / 4), kernel_size=3, dilation=5, padding=5)
        self.dconv4 = BasicConv2d(all_channel, int(all_channel / 4), kernel_size=3, dilation=7, padding=7)
        self.fuse_dconv = nn.Conv2d(all_channel, all_channel, kernel_size=3, padding=1)


    def forward(self, x, ir):
        multiplication = self.conv1(x * ir)
        summation = self.conv2(x + ir)
        fusion = (summation + multiplication)
        x1 = self.dconv1(fusion)
        x2 = self.dconv2(fusion)
        x3 = self.dconv3(fusion)
        x4 = self.dconv4(fusion)
        out = self.fuse_dconv(torch.cat((x1, x2, x3, x4), dim=1))

        return out


class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpSample, self).__init__()
        self.up_sample = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, 1, 1),
                                       nn.BatchNorm2d(out_channels),
                                       nn.GELU(),
                                       nn.UpsamplingBilinear2d(scale_factor=2, ))

    def forward(self, inputs):
        return self.up_sample(inputs)


class EdgeDet5(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EdgeDet5, self).__init__()

        self.cls_score = nn.Sequential(
            nn.Dropout2d(p=0.1),
            BasicConv2d(in_channels, in_channels // 2, kernel_size=3, padding=3, dilation=3),  # 256 -> 128
            UpSample(in_channels // 2, in_channels // 2),  # [14, 14] -> [28, 28]

            BasicConv2d(in_channels // 2, in_channels // 4, kernel_size=3, padding=3, dilation=3),  # 128 -> 64
            UpSample(in_channels // 4, in_channels // 4),  # [28, 28] -> [56, 56]

            BasicConv2d(in_channels // 4, in_channels // 8, kernel_size=3, padding=3, dilation=3),  # 64 -> 32
            UpSample(in_channels // 8, in_channels // 8),  # [56, 56] -> [128, 128]

            BasicConv2d(in_channels // 8, in_channels // 16, kernel_size=3, padding=3, dilation=3),  # 32 -> 16
            UpSample(in_channels // 16, in_channels // 16),  # [128, 128] -> [224, 224]

            nn.Conv2d(in_channels // 16, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        cls_score = self.cls_score(x)
        return cls_score


class EdgeDet3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EdgeDet3, self).__init__()

        self.cls_score = nn.Sequential(
            nn.Dropout2d(p=0.1),

            BasicConv2d(in_channels, in_channels // 2, kernel_size=3, padding=3, dilation=3),  # 128 -> 64
            UpSample(in_channels // 2, in_channels // 4),  # [56, 56] -> [128, 128]

            BasicConv2d(in_channels // 4, in_channels // 8, kernel_size=3, padding=3, dilation=3),  # 63 -> 32
            UpSample(in_channels // 8, in_channels // 8),  # [128, 128] -> [224, 224]

            nn.Conv2d(in_channels // 8, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        # print("at detect")
        cls_score = self.cls_score(x)
        return cls_score


class EdgeDet1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EdgeDet1, self).__init__()

        self.cls_score = nn.Sequential(
            nn.Dropout2d(p=0.1),
            BasicConv2d(in_channels, in_channels, kernel_size=3, padding=3, dilation=3),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        cls_score = self.cls_score(x)
        return cls_score


class PredictionDecoder(nn.Module):
    def __init__(self, channel1=64, channel2=128, channel3=256, channel4=256, channel5=512, out_channels=1,
                 ):
        super(PredictionDecoder, self).__init__()
        # 15 20
        self.decoder5 = nn.Sequential(
            nn.Dropout2d(p=0.1),
            BasicConv2d(channel5, channel5, kernel_size=3, padding=3, dilation=3),
            BasicConv2d(channel5, channel4, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        # 30 40
        self.decoder4 = nn.Sequential(
            nn.Dropout2d(p=0.1),
            BasicConv2d(channel4, channel4, kernel_size=3, padding=3, dilation=3),
            BasicConv2d(channel4, channel3, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        # 60 80
        self.decoder3 = nn.Sequential(
            nn.Dropout2d(p=0.1),
            BasicConv2d(channel3, channel3, kernel_size=3, padding=3, dilation=3),
            BasicConv2d(channel3, channel2, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        # 120 160
        self.decoder2 = nn.Sequential(
            nn.Dropout2d(p=0.1),
            BasicConv2d(channel2, channel2, kernel_size=3, padding=3, dilation=3),
            BasicConv2d(channel2, channel1, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        # self.semantic_pred2 = nn.Conv2d(channel1, n_classes, kernel_size=3, padding=1)
        # 240 320 -> 480 640
        self.decoder1 = nn.Sequential(
            nn.Dropout2d(p=0.1),
            BasicConv2d(channel1, channel1, kernel_size=3, padding=3, dilation=3),
            BasicConv2d(channel1, channel1, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),  # 480 640
            BasicConv2d(channel1, channel1, kernel_size=3, padding=1),
            nn.Conv2d(channel1, out_channels, kernel_size=3, padding=1)
        )

        self.edge_head5 = EdgeDet5(channel4, 1)
        self.edge_head3 = EdgeDet3(channel2, 1)
        self.edge_head1 = EdgeDet1(out_channels, 1)

    def forward(self, x5, x4, x3, x2, x1):
        x5_decoder = self.decoder5(x5)
        # for PST900 dataset since the input size is 720x1280, the size of x5_decoder and x4_decoder is 23 and 45,
        # so we cannot use 2x upsampling directrly. x5_decoder = F.interpolate(x5_decoder, size=fea_size,
        # mode="bilinear", align_corners=True)
        x4_decoder = self.decoder4(x5_decoder + x4)
        x3_decoder = self.decoder3(x4_decoder + x3)
        x2_decoder = self.decoder2(x3_decoder + x2)
        x1_decoder = self.decoder1(x2_decoder + x1)

        if self.training:
            """
            torch.Size([1, 256, 40, 30])
            torch.Size([1, 128, 160, 120])
            torch.Size([1, 1, 640, 480])
            """
            return [self.edge_head5(x5_decoder), self.edge_head3(x3_decoder), self.edge_head1(x1_decoder)]

        else:
            return self.edge_head1(x1_decoder)

# 消融实验： 去掉某个模块
# class delModule(nn.Module):
#     def __init__(self, in_channel, out_channel, kernel_size):
#         super().__init__()
#         self.conv = ConvLayer(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, stride = 1)

#     def forward(self, rgb, ir):
#         x = torch.cat((rgb, ir), dim = 1)
#         x = self.conv(x)
#         return x
        

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        (
            self.layer1_rgb,
            self.layer2_rgb,
            self.layer3_rgb,
            self.layer4_rgb,
            self.layer5_rgb,
        ) = Backbone_ResNet152_in3(pretrained=True)

        # reduce the channel number, input: 480 640
        self.rgbconv1 = BasicConv2d(64, 64, kernel_size=3, padding=1)  # 240 320
        self.rgbconv2 = BasicConv2d(256, 128, kernel_size=3, padding=1)  # 120 160
        self.rgbconv3 = BasicConv2d(512, 256, kernel_size=3, padding=1)  # 60 80
        self.rgbconv4 = BasicConv2d(1024, 256, kernel_size=3, padding=1)  # 30 40
        self.rgbconv5 = BasicConv2d(2048, 512, kernel_size=3, padding=1)  # 15 20

        # self.rgbconv1 = BasicConv2d(64, 64, kernel_size=3, padding=1)  # 240 320
        # self.rgbconv2 = BasicConv2d(64, 128, kernel_size=3, padding=1)  # 120 160
        # self.rgbconv3 = BasicConv2d(128, 256, kernel_size=3, padding=1)  # 60 80
        # self.rgbconv4 = BasicConv2d(256, 256, kernel_size=3, padding=1)  # 30 40
        # self.rgbconv5 = BasicConv2d(512, 512, kernel_size=3, padding=1)  # 15 20

        self.RFN5 = CRF(512)
        self.SCA4 = SCA(256)
        self.SCA3 = SCA(256)
        self.SCA2 = SCA(128)
        self.EDS1 = EDS(64)


        self.decoder = PredictionDecoder(64, 128, 256, 256, 512, 1)

    def forward(self, rgb, ir):
        x1 = self.layer1_rgb(rgb)
        x2 = self.layer2_rgb(x1)
        x3 = self.layer3_rgb(x2)
        x4 = self.layer4_rgb(x3)
        x5 = self.layer5_rgb(x4)

        ir1 = self.layer1_rgb(ir)
        ir2 = self.layer2_rgb(ir1)
        ir3 = self.layer3_rgb(ir2)
        ir4 = self.layer4_rgb(ir3)
        ir5 = self.layer5_rgb(ir4)

        x1 = self.rgbconv1(x1)
        x2 = self.rgbconv2(x2)
        x3 = self.rgbconv3(x3)
        x4 = self.rgbconv4(x4)
        x5 = self.rgbconv5(x5)

        ir1 = self.rgbconv1(ir1)
        ir2 = self.rgbconv2(ir2)
        ir3 = self.rgbconv3(ir3)
        ir4 = self.rgbconv4(ir4)
        ir5 = self.rgbconv5(ir5)


        out5 = self.RFN5(x5, ir5)
        out4 = self.SCA4(x4, ir4)
        out3 = self.SCA3(x3, ir3)
        out2 = self.SCA2(x2, ir2)
        out1 = self.EDS1(x1, ir1)

        return self.decoder(out5, out4, out3, out2, out1)


if __name__ == '__main__':
    model = Net()
    rgb = torch.randn(size=(1, 3, 640, 480))
    t = torch.randn(size=(1, 3, 640, 480))
    out = model(rgb, t)
    for o in out:
        print(o.shape)
