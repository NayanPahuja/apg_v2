import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, channels, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(channels, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x
    
class AquaPixGAN_Nets:
    def __init__(self, base_model='pix2pix'):
        if base_model=='pix2pix': # default
            self.netG = GeneratorAquaPixGan() 
            self.netD = DiscriminatorAquaPixGan()
        elif base_model=='resnet':
            #TODO: add ResNet support
            pass
        else: 
            pass

class GeneratorAquaPixGan(nn.Module):
    def __init__(self):
        super(GeneratorAquaPixGan, self).__init__()
    
        self.e1 = nn.Sequential(_EncodeLayer(3, 64, batch_normalize=False), CBAM(64))
        self.e2 = nn.Sequential(_EncodeLayer(64, 128), CBAM(128))
        self.e3 = nn.Sequential(_EncodeLayer(128, 256), CBAM(256))
        self.e4 = nn.Sequential(_EncodeLayer(256, 512), CBAM(512))
        self.e5 = nn.Sequential(_EncodeLayer(512, 512), CBAM(512))
        self.e6 = nn.Sequential(_EncodeLayer(512, 512), CBAM(512))
        self.e7 = nn.Sequential(_EncodeLayer(512, 512), CBAM(512))
        self.e8 = nn.Sequential(_EncodeLayer(512, 512), CBAM(512))
        
        self.d1 = _DecodeLayer(512, 512, dropout=True)
        self.d2 = _DecodeLayer(1024, 512, dropout=True)
        self.d3 = _DecodeLayer(1024, 512, dropout=True)
        self.d4 = _DecodeLayer(1024, 512)
        self.d5 = _DecodeLayer(1024, 256)
        self.d6 = _DecodeLayer(512, 128)
        self.d7 = _DecodeLayer(256, 64)

        self.cbam_d1 = CBAM(1024)
        self.cbam_d2 = CBAM(1024)
        self.cbam_d3 = CBAM(1024)
        self.cbam_d4 = CBAM(1024)
        self.cbam_d5 = CBAM(512)
        self.cbam_d6 = CBAM(256)
        self.cbam_d7 = CBAM(128)

        self.deconv = nn.ConvTranspose2d(
            in_channels=128, out_channels=3, 
            kernel_size=4, stride=2, padding=1, bias=False
        )

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        e5 = self.e5(e4)
        e6 = self.e6(e5)
        e7 = self.e7(e6)
        e8 = self.e8(e7)

        d1 = self.cbam_d1(self.d1(e8, e7))
        d2 = self.cbam_d2(self.d2(d1, e6))
        d3 = self.cbam_d3(self.d3(d2, e5))
        d4 = self.cbam_d4(self.d4(d3, e4))
        d5 = self.cbam_d5(self.d5(d4, e3))
        d6 = self.cbam_d6(self.d6(d5, e2))
        d7 = self.cbam_d7(self.d7(d6, e1))

        final = self.deconv(d7)

        return final

class _EncodeLayer(nn.Module):
    """ Encoder: a series of Convolution-BatchNorm-ReLU*
    """
    def __init__(self, in_size, out_size, batch_normalize=True):
        super(_EncodeLayer, self).__init__()
        layers = [nn.Conv2d(in_channels=in_size, out_channels=out_size, kernel_size=4, stride=2, padding=1, bias=False)]
        if batch_normalize: 
            layers.append(nn.BatchNorm2d(num_features=out_size))
        layers.append(nn.LeakyReLU(negative_slope=0.2))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class _DecodeLayer(nn.Module):
    """ Decoder: a series of Convolution-BatchNormDropout-ReLU*
    """
    def __init__(self, in_size, out_size, dropout=False):
        super(_DecodeLayer, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_channels=in_size, out_channels=out_size, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=out_size),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(0.5))
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)
        return x

class DiscriminatorAquaPixGan(nn.Module):
    def __init__(self, in_channels=3):
        super(DiscriminatorAquaPixGan, self).__init__()

        def down_layer(in_filters, out_filters, normalization=True):
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.BatchNorm2d(out_filters, momentum=0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(CBAM(out_filters))
            return layers

        self.model = nn.Sequential(
            *down_layer(2*in_channels, 64, normalization=False),
            *down_layer(64, 128),
            *down_layer(128, 256),
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 512, 4, padding=0, bias=False),
            nn.BatchNorm2d(512, momentum=0.8),
            CBAM(512),
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 1, 4, padding=0, bias=False)
        )

    def forward(self, img_A, img_B):
        img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)