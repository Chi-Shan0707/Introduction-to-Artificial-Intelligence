import torch

import torch.nn as nn


class VGGBlock(nn.Moudle):

    def __init__(self, in_channels, out_channels):

        super().__init__()

        self.conv1 =nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)



        self.relu2 = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x =self.relu1(self.conv1(x))

        x = self.relu2(self.conv2(x))

        x = self.pool(x)

        return x
    
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        
        super().__init__()
        
        
        self.bn2 = nn.BatchNorm2d(channels)


