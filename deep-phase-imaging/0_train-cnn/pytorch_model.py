import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def imresize_big(img, factor):
  '''
  :param img: mask
  :param factor: scale factor
  :return: resize img
  '''
  img_big = np.zeros((img.shape[0]*factor, img.shape[1]*factor))
  for i in range(0, img.shape[0]):
    for j in range(0, img.shape[1]):
      x = i*factor
      y = j*factor
      for a in range(0, factor):
        for b in range(0, factor):
          img_big[x+a, y+b] = img[i, j]
  return img_big


class ResidualBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(input_dim, output_dim, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(input_dim)
        self.conv2 = nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(output_dim)
        self.conv_residual = nn.Conv2d(input_dim, output_dim, kernel_size=3, padding=1)

    def forward(self, x):
        residual = x
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv1(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.conv2(x)
        residual = self.conv_residual(residual)
        x += residual
        return x


class ResidualDownsamplingBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(input_dim, output_dim, kernel_size=3, padding=1, stride=2)
        self.bn1 = nn.BatchNorm2d(input_dim)
        self.conv2 = nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(output_dim)
        self.conv_residual = nn.Conv2d(input_dim, output_dim, kernel_size=3, padding=1, stride=2)

    def forward(self, x):
        residual = x
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv1(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.conv2(x)
        residual = self.conv_residual(residual)
        x += residual
        return x


class ResidualUpsamplingBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.output_dim = output_dim
        self.bn1 = nn.BatchNorm2d(input_dim)
        self.conv1 = nn.ConvTranspose2d(input_dim, output_dim, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(output_dim)
        self.conv2 = nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1)
        self.conv_residual = nn.ConvTranspose2d(input_dim, output_dim, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, x):
        residual = x
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv1(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.conv2(x)
        residual = self.conv_residual(residual)
        x += residual
        return x


class ResidualUNet(nn.Module):
    def __init__(self, dr_rate):
        super().__init__()
        self.dr_rate = dr_rate
        self.conv_input = ResidualBlock(2, 16)
        self.conv_input_cna = nn.Dropout(p=self.dr_rate)

        # left
        self.down_conv_1 = ResidualDownsamplingBlock(16, 32)
        self.down_conv_1_cna = nn.Dropout(p=self.dr_rate)

        self.down_conv_2 = ResidualDownsamplingBlock(32, 64)
        self.down_conv_2_cna = nn.Dropout(p=self.dr_rate)

        self.down_conv_3 = ResidualDownsamplingBlock(64, 128)
        self.down_conv_3_cna = nn.Dropout(p=self.dr_rate)

        # center
        self.center_conv = ResidualDownsamplingBlock(128, 256)

        # right
        self.up_conv_1 = ResidualUpsamplingBlock(256, 128)
        self.up_conv_1_dp = nn.Dropout(p=self.dr_rate)

        self.up_conv_2 = ResidualUpsamplingBlock(256, 128)
        self.up_conv_2_dp = nn.Dropout(p=self.dr_rate)

        self.up_conv_3 = ResidualUpsamplingBlock(192, 64)
        self.up_conv_3_dp = nn.Dropout(p=self.dr_rate)

        self.up_conv_4 = ResidualUpsamplingBlock(96, 32)
        self.up_conv_4_dp = nn.Dropout(p=self.dr_rate)

        self.conv = ResidualBlock(48, 32)
        self.conv_dp = nn.Dropout(p=self.dr_rate)

        self.conv_output = ResidualBlock(32, 1)

    def forward(self, x):
        x = self.conv_input(x)
        x_conv_input_cna = self.conv_input_cna(x)
        x = self.down_conv_1(x_conv_input_cna)
        x_down_conv_1_cna = self.down_conv_1_cna(x)
        x = self.down_conv_2(x_down_conv_1_cna)
        x_down_conv_2_cna = self.down_conv_2_cna(x)
        x = self.down_conv_3(x_down_conv_2_cna)
        x_down_conv_3_cna = self.down_conv_3_cna(x)

        x = self.center_conv(x_down_conv_3_cna)

        x = self.up_conv_1(x)
        x = self.up_conv_1_dp(x)
        temp1 = torch.cat((x, x_down_conv_3_cna), dim=1)
        x = self.up_conv_2(temp1)
        x = self.up_conv_2_dp(x)
        temp2 = torch.cat((x, x_down_conv_2_cna), dim=1)
        x = self.up_conv_3(temp2)
        x = self.up_conv_3_dp(x)
        temp3 = torch.cat((x, x_down_conv_1_cna), dim=1)
        x = self.up_conv_4(temp3)
        x = self.up_conv_4_dp(x)
        temp4 = torch.cat((x, x_conv_input_cna), dim=1)
        x = self.conv(temp4)
        x = self.conv_output(x)

        return x

