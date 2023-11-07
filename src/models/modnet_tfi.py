import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .modules import TFI, SBFI, DBFI

from .backbones import SUPPORTED_BACKBONES


#------------------------------------------------------------------------------
#  MODNet Basic Modules
#------------------------------------------------------------------------------

class IBNorm(nn.Module):
    """ Combine Instance Norm and Batch Norm into One Layer
    """

    def __init__(self, in_channels):
        super(IBNorm, self).__init__()
        in_channels = in_channels
        self.bnorm_channels = int(in_channels / 2)
        self.inorm_channels = in_channels - self.bnorm_channels

        self.bnorm = nn.BatchNorm2d(self.bnorm_channels, affine=True)
        self.inorm = nn.InstanceNorm2d(self.inorm_channels, affine=False)
        
    def forward(self, x):
        bn_x = self.bnorm(x[:, :self.bnorm_channels, ...].contiguous())
        in_x = self.inorm(x[:, self.bnorm_channels:, ...].contiguous())

        return torch.cat((bn_x, in_x), 1)


class Conv2dIBNormRelu(nn.Module):
    """ Convolution + IBNorm + ReLu
    """

    def __init__(self, in_channels, out_channels, kernel_size, 
                 stride=1, padding=0, dilation=1, groups=1, bias=True, 
                 with_ibn=True, with_relu=True):
        super(Conv2dIBNormRelu, self).__init__()

        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, 
                      stride=stride, padding=padding, dilation=dilation, 
                      groups=groups, bias=bias)
        ]

        if with_ibn:       
            layers.append(IBNorm(out_channels))
        if with_relu:
            layers.append(nn.ReLU(inplace=True))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x) 


class SEBlock(nn.Module):
    """ SE Block Proposed in https://arxiv.org/pdf/1709.01507.pdf 
    """

    def __init__(self, in_channels, out_channels, reduction=1):
        super(SEBlock, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, int(in_channels // reduction), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(int(in_channels // reduction), out_channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)

        return x * w.expand_as(x)


#------------------------------------------------------------------------------
#  MODNet
#------------------------------------------------------------------------------

class mergedDecoder(nn.Module):
    """ Low Resolution Branch of MODNet
       """

    def __init__(self, backbone,hr_channels, enc_channels):
        super(mergedDecoder, self).__init__()

        """ low Resolution Branch of MODNet
            """
        enc_channels = backbone.enc_channels

        self.backbone = backbone
        # self.enc_channels = [16, 24, 32, 96, 1280]
        self.se_block = SEBlock(enc_channels[4], enc_channels[4], reduction=4)
        self.se_block2 = SEBlock(enc_channels[2], enc_channels[2], reduction=4)
        self.conv_lr16x = Conv2dIBNormRelu(enc_channels[4], enc_channels[3], 5, stride=1, padding=2)
        self.conv_lr16x2 = Conv2dIBNormRelu(enc_channels[2], enc_channels[1], 5, stride=1, padding=2)
        self.conv_lr8x = Conv2dIBNormRelu(enc_channels[3], enc_channels[2], 5, stride=1, padding=2)
        self.conv_lr8x2 = Conv2dIBNormRelu(enc_channels[1], enc_channels[0], 5, stride=1, padding=2)
        self.conv_lr = Conv2dIBNormRelu(enc_channels[2], 1, kernel_size=3, stride=2, padding=1, with_ibn=False,
                                        with_relu=False)
        self.conv_l2 = Conv2dIBNormRelu(enc_channels[0], 1, kernel_size=3, stride=2, padding=1, with_ibn=False,
                                        with_relu=False)
        """ High Resolution Branch of MODNet
            """
        self.tohr_enc2x = Conv2dIBNormRelu(enc_channels[0], hr_channels, 1, stride=1, padding=0)
        self.conv_enc2x = Conv2dIBNormRelu(hr_channels + 3, hr_channels, 3, stride=2, padding=1)

        self.tohr_enc4x = Conv2dIBNormRelu(enc_channels[1], hr_channels, 1, stride=1, padding=0)
        self.conv_enc4x = Conv2dIBNormRelu(2 * hr_channels, 2 * hr_channels, 3, stride=1, padding=1)

        self.conv_hr4x = nn.Sequential(
            #32 99
            #Conv2dIBNormRelu(3 * hr_channels + 3, 2 * hr_channels, 3, stride=1, padding=1),
            Conv2dIBNormRelu(hr_channels + 3, 2 * hr_channels, 3, stride=1, padding=1),
            Conv2dIBNormRelu(2 * hr_channels, 2 * hr_channels, 3, stride=1, padding=1),
            Conv2dIBNormRelu(2 * hr_channels, hr_channels, 3, stride=1, padding=1),
        )

        self.conv_hr2x = nn.Sequential(
            Conv2dIBNormRelu(2 * hr_channels, 2 * hr_channels, 3, stride=1, padding=1),
            Conv2dIBNormRelu(2 * hr_channels, hr_channels, 3, stride=1, padding=1),
            Conv2dIBNormRelu(hr_channels, hr_channels, 3, stride=1, padding=1),
            Conv2dIBNormRelu(hr_channels, hr_channels, 3, stride=1, padding=1),
        )

        self.conv_hr = nn.Sequential(
            Conv2dIBNormRelu(hr_channels + 3, hr_channels, 3, stride=1, padding=1),
            Conv2dIBNormRelu(hr_channels, 1, kernel_size=1, stride=1, padding=0, with_ibn=False, with_relu=False),
        )


        self.tfi_0 = TFI(32,24,64)
        
        self.sbfi_0 = SBFI(64, 32, 2)

        self.dbfi_0 = DBFI(16, 16, 8)

    def forward(self, img, inference):
        # img of 512x512
        enc_features = self.backbone.forward(img) #encoder
        enc2x, enc4x, enc8x, enc32x = enc_features[0], enc_features[1],enc_features[2], enc_features[4]  # 4 sets of encoder embeddings
        # print(np.shape(img))#512
        # print(np.shape(enc2x))#256
        # print(np.shape(enc4x))#128-24
        # print(np.shape(enc8x))#64
        # print(np.shape(enc16x))#32
        # print(np.shape(enc32x))#16
        
        # low-resolution branch
        # e-ASPP      
        # enc32x
        enc32x = self.se_block(enc32x)#torch.Size([8, 1280, 16, 16])
        lr16x = F.interpolate(enc32x, scale_factor=2, mode='bilinear', align_corners=False)#torch.Size([8, 1280, 32, 32])
        lr16x = self.conv_lr16x(lr16x)#torch.Size([8, 96, 32, 32])
        lr8x = F.interpolate(lr16x, scale_factor=2, mode='bilinear', align_corners=False)#torch.Size([8, 32, 64, 64])
        lr8x = self.conv_lr8x(lr8x)#torch.Size([8, 32, 64, 64])

        # enc8x
        enc8x = self.se_block2(enc8x)#64
        lr4x = F.interpolate(enc8x, scale_factor=2, mode='bilinear', align_corners=False)#torch.Size([8, 32, 128, 128])
        lr4x = self.conv_lr16x2(lr4x)#torch.Size([8, 24, 128, 128])
        lr2x = F.interpolate(lr4x, scale_factor=2, mode='bilinear', align_corners=False)#torch.Size([8, 24, 256, 256])
        lr2x = self.conv_lr8x2(lr2x)#torch.Size([8, 16, 256, 256])

        pred_semantic = None
        if not inference:
            lr = self.conv_lr(lr8x)
            pred_semantic = torch.sigmoid(lr)#torch.Size([8, 1, 32, 32])

        # return pred_semantic, lr8x, [enc2x, enc4x]
        
        # high-resolution branch
        # def forward(self, img, enc2x, enc4x, lr8x, inference)
        
        img2x = F.interpolate(img, scale_factor=1 / 2, mode='bilinear', align_corners=False)#torch.Size([8, 3, 256, 256])
        img4x = F.interpolate(img, scale_factor=1 / 4, mode='bilinear', align_corners=False)#torch.Size([8, 3, 128, 128])

        enc2x = self.tohr_enc2x(enc2x)#torch.Size([8, 32, 256, 256]),encoder feat
        hr4x = self.conv_enc2x(torch.cat((img2x, enc2x), dim=1))#torch.Size([8, 32, 128, 128]),enc feat+img feat

        enc4x = self.tohr_enc4x(enc4x)#torch.Size([8, 32, 128, 128]) enc feat
        hr4x = self.conv_enc4x(torch.cat((hr4x, enc4x), dim=1))#torch.Size([8, 64, 128, 128]) 2 enc + img feat
        
        tfi_layer = self.tfi_0(enc4x,lr4x,hr4x)#torch.Size([8, 32, 128, 128])

        #lr4x = F.interpolate(lr8x, scale_factor=2, mode='bilinear', align_corners=False)#torch.Size([8, 32, 128, 128]),lb feat
        #hr4x = self.conv_hr4x(torch.cat((hr4x, lr4x, img4x), dim=1))#torch.Size([8, 32, 128, 128]),3 enc feat+1 lb feat+ 2 img feat

        hr4x = self.conv_hr4x(torch.cat((tfi_layer, img4x), dim=1))
        
        hr2x = F.interpolate(hr4x, scale_factor=2, mode='bilinear', align_corners=False)#torch.Size([8, 32, 256, 256])
        hr2x = self.conv_hr2x(torch.cat((hr2x, enc2x), dim=1))#torch.Size([8, 32, 256, 256])

        pred_detail = None
        if not inference:
            hr = F.interpolate(hr2x, scale_factor=2, mode='bilinear', align_corners=False)
            hr = self.conv_hr(torch.cat((hr, img), dim=1))
            pred_detail = torch.sigmoid(hr)

        return pred_semantic,pred_detail,lr8x,hr2x


class FusionBranch(nn.Module):
    """ Fusion Branch of MODNet
    """

    def __init__(self, hr_channels, enc_channels):
        super(FusionBranch, self).__init__()
        self.conv_lr4x = Conv2dIBNormRelu(enc_channels[2], hr_channels, 5, stride=1, padding=2)
        
        self.conv_f2x = Conv2dIBNormRelu(2 * hr_channels, hr_channels, 3, stride=1, padding=1)
        self.conv_f = nn.Sequential(
            Conv2dIBNormRelu(hr_channels + 3, int(hr_channels / 2), 3, stride=1, padding=1),
            Conv2dIBNormRelu(int(hr_channels / 2), 1, 1, stride=1, padding=0, with_ibn=False, with_relu=False),
        )

    def forward(self, img, lr8x, hr2x):
        lr4x = F.interpolate(lr8x, scale_factor=2, mode='bilinear', align_corners=False)
        lr4x = self.conv_lr4x(lr4x)
        lr2x = F.interpolate(lr4x, scale_factor=2, mode='bilinear', align_corners=False)

        f2x = self.conv_f2x(torch.cat((lr2x, hr2x), dim=1))
        f = F.interpolate(f2x, scale_factor=2, mode='bilinear', align_corners=False)
        f = self.conv_f(torch.cat((f, img), dim=1))
        pred_matte = torch.sigmoid(f)

        return pred_matte


#------------------------------------------------------------------------------
#  MODNet
#------------------------------------------------------------------------------

class MODNet(nn.Module):
    """ Architecture of MODNet
    """

    def __init__(self, in_channels=3, hr_channels=32, backbone_arch='mobilenetv2', backbone_pretrained=True):
        super(MODNet, self).__init__()

        self.in_channels = in_channels
        self.hr_channels = hr_channels
        self.backbone_arch = backbone_arch
        self.backbone_pretrained = backbone_pretrained

        self.backbone = SUPPORTED_BACKBONES[self.backbone_arch](self.in_channels)

        self.merged_decoder = mergedDecoder(self.backbone,self.hr_channels, self.backbone.enc_channels)
        self.f_branch = FusionBranch(self.hr_channels, self.backbone.enc_channels)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                self._init_conv(m)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.InstanceNorm2d):
                self._init_norm(m)

        if self.backbone_pretrained:
            self.backbone.load_pretrained_ckpt()                

    def forward(self, img, inference):
        pred_semantic, pred_detail, lr8x, hr2x = self.merged_decoder(img, inference)
        pred_matte = self.f_branch(img, lr8x, hr2x)

        return pred_semantic, pred_detail, pred_matte
    
    def freeze_norm(self):
        norm_types = [nn.BatchNorm2d, nn.InstanceNorm2d]
        for m in self.modules():
            for n in norm_types:
                if isinstance(m, n):
                    m.eval()
                    continue

    def _init_conv(self, conv):
        nn.init.kaiming_uniform_(
            conv.weight, a=0, mode='fan_in', nonlinearity='relu')
        if conv.bias is not None:
            nn.init.constant_(conv.bias, 0)

    def _init_norm(self, norm):
        if norm.weight is not None:
            nn.init.constant_(norm.weight, 1)
            nn.init.constant_(norm.bias, 0)
