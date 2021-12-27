import torch
import torch.nn as nn
import torch.nn.functional as F


class conv_block(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_ch, out_ch, stride=1, padding=1):
        super(conv_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=padding, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_ch))

    def forward(self, x):
        x = self.conv(x)
        return x


class conv_T_block(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, padding=1) -> None:
        super(conv_T_block, self).__init__()
        self.conv_t = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=stride, padding=padding),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_ch))

    def forward(self, x):
        x = self.conv_t(x)
        return x


class Simple_Unet(torch.nn.Module):
    def __init__(self, in_ch=3, out_ch=1) -> None:
        super(Simple_Unet, self).__init__()
        self.conv1_1 = conv_block(in_ch, 9, 2, 1)
        self.conv1_2 = conv_block(9, 12, 2, 1)
        self.conv1_3 = conv_block(12, 12, 2, 1)

        self.up2_1 = nn.Upsample(scale_factor=2)
        self.conv_t2_1 = conv_T_block(24, 9)

        self.up2_2 = nn.Upsample(scale_factor=2)
        self.conv_t2_2 = conv_T_block(18, 12)

        self.up2_3 = nn.Upsample(scale_factor=2)
        self.conv_t2_3 = conv_T_block(15, 1)

    def forward(self, x):
        skips = []
        x = F.pad(x, (16, 16, 1, 1), "constant", 0)
        skips.append(x)
        # print(f"padding: {x.shape}")
        x = self.conv1_1(x)
        skips.append(x)
        # print(f"conv1_1: {x.shape}")
        x = self.conv1_2(x)
        skips.append(x)
        # print(f"conv1_2: {x.shape}")
        x = self.conv1_3(x)

        x = self.up2_1(x)
        # print(f"up2_1: {x.shape}")
        x = torch.cat((x, skips.pop()), dim=-3)
        x = self.conv_t2_1(x)

        x = self.up2_2(x)
        # print(f"up2_2: {x.shape}")
        x = torch.cat((x, skips.pop()), dim=-3)
        x = self.conv_t2_2(x)

        x = self.up2_3(x)
        # print(f"up2_3: {x.shape}")
        x = torch.cat((x, skips.pop()), dim=-3)
        x = self.conv_t2_3(x)

        x = F.pad(x, (-16, -16, -1, -1))
        return torch.sigmoid(x)
