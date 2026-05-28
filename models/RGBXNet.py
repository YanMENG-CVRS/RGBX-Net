import torch
import torch.nn as nn
import torch.nn.functional as F

from models.FusionModule import FeatureFusionBlock
from models.MLPDecoder import DecoderHead
from models.swin import swin_b


class DepthEncoder(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DepthEncoder, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        return x


class RGBXNetBackbone(nn.Module):
    def __init__(self, dim, n_class, x_dim=3):
        super().__init__()
        self.encoder2 = swin_b(pretrained=True)
        self.d_encoders = nn.ModuleList(
            [
                DepthEncoder(x_dim, 64),
                DepthEncoder(64, 128),
                DepthEncoder(128, 256),
                DepthEncoder(256, 512),
                DepthEncoder(512, 1024),
            ]
        )
        self.ffb1 = FeatureFusionBlock(dim=dim * 1)
        self.ffb2 = FeatureFusionBlock(dim=dim * 2)
        self.ffb3 = FeatureFusionBlock(dim=dim * 4)
        self.ffb4 = FeatureFusionBlock(dim=dim * 8)
        self.head = DecoderHead(num_classes=n_class)

    def forward(self, x, xx):
        out = self.encoder2(x)
        swin_b1, swin_b2, swin_b3, swin_b4 = out[0], out[1], out[2], out[3]

        for i in range(2):
            xx = self.d_encoders[i](xx)

        swin_b1, ld1 = self.ffb1(swin_b1, xx)
        ld2 = self.d_encoders[2](ld1)
        swin_b2, ld2 = self.ffb2(swin_b2, ld2)
        ld3 = self.d_encoders[3](ld2)
        swin_b3, ld3 = self.ffb3(swin_b3, ld3)
        ld4 = self.d_encoders[4](ld3)
        swin_b4, ld4 = self.ffb4(swin_b4, ld4)

        return self.head([swin_b1, swin_b2, swin_b3, swin_b4])


class EncoderDecoder(nn.Module):
    def __init__(self, criterion=nn.CrossEntropyLoss(reduction="mean", ignore_index=255), num_classes=2, x_dim=3):
        super(EncoderDecoder, self).__init__()
        self.backbone = RGBXNetBackbone(128, num_classes, x_dim)
        self.criterion = criterion

    def encode_decode(self, rgb, modal_x):
        ori_size = rgb.shape
        out = self.backbone(rgb, modal_x)
        out = F.interpolate(out, size=ori_size[-2:], mode="bilinear", align_corners=False)
        return out

    def forward(self, rgb, modal_x=None, label=None):
        out = self.encode_decode(rgb, modal_x)
        if label is not None:
            return self.criterion(out, label.long())
        return out


EncoderDecoderV8 = EncoderDecoder
