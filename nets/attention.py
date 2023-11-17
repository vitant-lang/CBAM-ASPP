
import math
import torch
import torch.nn as nn
class se_block(nn.Module):
    def __init__(self, channel, ratio=16):
        super(se_block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // ratio, channel, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.fc(y)
        return x * y





class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio=16):
        super(ChannelAttention, self).__init__()
        self.in_channels = in_channels  # 添加这一行
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels * 2, in_channels // ratio),
            nn.ReLU(),
            nn.Linear(in_channels // ratio, in_channels)
        )

    def forward(self, x):
        avg_out = self.avg_pool(x)
        max_out = self.max_pool(x)

        avg_max_out = torch.cat([avg_out, max_out], dim=1).view(x.size(0), -1)

        channel_attention = torch.sigmoid(self.fc(avg_max_out)).view(x.size(0), self.in_channels, 1, 1)  # 使用 self.in_channels

        return x * channel_attention

class SpatialAttention(nn.Module):
    def __init__(self, in_channels, kernel_size=7):  # 添加 in_channels 参数
        super(SpatialAttention, self).__init__()
        self.in_channels = in_channels  # 添加这一行
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=(kernel_size-1)//2, bias=False)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        spatial_attention = torch.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))

        return x * spatial_attention

class CBAM(nn.Module):
    def __init__(self, in_channels, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, ratio)
        self.spatial_attention = SpatialAttention(in_channels, kernel_size)  # 传入 in_channels 参数

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class CBAM(nn.Module):
    def __init__(self, in_channels, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x




class eca_block(nn.Module):
    def __init__(self, channel, b=1, gamma=2):
        super(eca_block, self).__init__()
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)
