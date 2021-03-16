import torch
import torch.nn as nn
import torch.nn.functional as F


def interpolate_like(src, tar, mode='bilinear'):
    """
    Billinear interpolation of src to match the dimensionality of tar.
    """
    src = F.interpolate(src, size=tar.shape[2:], mode=mode, align_corners=True)
    
    return src


class DoubleConv(nn.Module):
    """
    Vanilla UNet block with optional dropout.
    """
    def __init__(self, in_channels, out_channels, pool=False, dropout=False):
        super(DoubleConv, self).__init__()

        modules = [
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]

        if pool:
            modules.insert(0, nn.MaxPool2d(2, 2))

        if dropout:
            modules.append(nn.Dropout2d(p=0.20))

        self.block = nn.Sequential(*modules)


    def forward(self, x):
        x = self.block(x)
        
        return x


class UNet(nn.Module):
    """
    Basic UNet architecture that takes a Sentinel-1 patch as input and injects AMSR and DST in the last ConvBlock.
    """

    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        
        filters = [16, 32, 64, 128, 256]

        self.enc1 = DoubleConv(in_channels, filters[0], dropout=True)
        self.enc2 = DoubleConv(filters[0], filters[1], pool=True, dropout=True)
        self.enc3 = DoubleConv(filters[1], filters[2], pool=True, dropout=True)
        self.enc4 = DoubleConv(filters[2], filters[3], pool=True, dropout=True)
        self.enc5 = DoubleConv(filters[3], filters[4], pool=True, dropout=True)

        self.dec4 = DoubleConv(filters[4] + filters[3], filters[3])
        self.dec3 = DoubleConv(filters[3] + filters[2], filters[2])
        self.dec2 = DoubleConv(filters[2] + filters[1], filters[1])
        self.dec1 = DoubleConv(filters[1] + filters[0] + 1 + 14, filters[0])

        self.out = nn.Conv2d(filters[0], out_channels, kernel_size=1)

    
    def forward(self, S1, DST, AMSR):

        enc1 = self.enc1(S1)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        enc5 = self.enc5(enc4)

        x = interpolate_like(enc5, enc4)
        x = self.dec4(torch.cat([x, enc4], dim=1))
        x = interpolate_like(x, enc3)
        x = self.dec3(torch.cat([x, enc3], dim=1))
        x = interpolate_like(x, enc2)
        x = self.dec2(torch.cat([x, enc2], dim=1))
        x = interpolate_like(x, enc1)
        x = self.dec1(torch.cat([x, enc1, DST, AMSR], dim=1))

        x = self.out(x)

        return x


# sanity check
if __name__ == "__main__":
    from pytorch_model_summary import summary

    patch_size = 300
    net = UNet(in_channels=2, out_channels=11)
    S1 = torch.rand(1, 2, patch_size, patch_size)
    DST = torch.rand(1, 1, patch_size, patch_size)
    AMSR = torch.rand(1, 14, patch_size, patch_size)

    print(summary(net, *[S1, DST, AMSR]))    