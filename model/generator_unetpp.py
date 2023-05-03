import torch
import torch.nn as nn
import torch.nn.functional as F

from config import global_config
CFG = global_config.cfg


class GeneratorNode(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(negative_slope=CFG.TRAIN.RELU_GAMA)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out


# 生成器定义
class Generator(nn.Module):
    def __init__(self, image_size, input_channels=3):
        super().__init__()

        # 生成器特征层数量
        gen_filter = [32, 64, 128, 256, 512]
        # 池化操作
        self.pool_2x = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool_4x = nn.MaxPool2d(kernel_size=4, stride=4)
        # 上采样操作
        # self.up_2x = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # self.up_4x = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        # Unet++ L1
        self.conv0_0 = GeneratorNode(input_channels, gen_filter[0])
        self.conv1_0 = GeneratorNode(gen_filter[0], gen_filter[1])
        self.conv0_1 = GeneratorNode(gen_filter[0] + gen_filter[1], gen_filter[0])
        self.up1_0 = nn.ConvTranspose2d(gen_filter[1], gen_filter[1], kernel_size=2, stride=2, padding=0)
        # Unet++ L2
        self.conv2_0 = GeneratorNode(gen_filter[1], gen_filter[2])
        self.conv1_1 = GeneratorNode(gen_filter[1] + gen_filter[2], gen_filter[1])
        self.conv0_2 = GeneratorNode(gen_filter[0] * 2 + gen_filter[1], gen_filter[0])
        self.up2_0 = nn.ConvTranspose2d(gen_filter[2], gen_filter[2], kernel_size=2, stride=2, padding=0)
        self.up1_1 = nn.ConvTranspose2d(gen_filter[1], gen_filter[1], kernel_size=2, stride=2, padding=0)
        # Unet++ L3
        self.conv3_0 = GeneratorNode(gen_filter[2], gen_filter[3])
        self.conv2_1 = GeneratorNode(gen_filter[2] + gen_filter[3], gen_filter[2])
        self.conv1_2 = GeneratorNode(gen_filter[1] * 2 + gen_filter[2], gen_filter[1])
        self.conv0_3 = GeneratorNode(gen_filter[0] * 3 + gen_filter[1], gen_filter[0])
        self.up3_0 = nn.ConvTranspose2d(gen_filter[3], gen_filter[3], kernel_size=4, stride=4, padding=0)
        self.up2_1 = nn.ConvTranspose2d(gen_filter[2], gen_filter[2], kernel_size=2, stride=2, padding=0)
        self.up1_2 = nn.ConvTranspose2d(gen_filter[1], gen_filter[1], kernel_size=2, stride=2, padding=0)
        # Unet++ L4
        self.conv4_0 = GeneratorNode(gen_filter[3], gen_filter[4])
        self.conv3_1 = GeneratorNode(gen_filter[3] + gen_filter[4], gen_filter[3])
        self.conv2_2 = GeneratorNode(gen_filter[2] * 2 + gen_filter[3], gen_filter[2])
        self.conv1_3 = GeneratorNode(gen_filter[1] * 3 + gen_filter[2], gen_filter[1])
        self.conv0_4 = GeneratorNode(gen_filter[0] * 4 + gen_filter[1], gen_filter[0])
        self.up4_0 = nn.ConvTranspose2d(gen_filter[4], gen_filter[4], kernel_size=4, stride=4, padding=0)
        self.up3_1 = nn.ConvTranspose2d(gen_filter[3], gen_filter[3], kernel_size=4, stride=4, padding=0)
        self.up2_2 = nn.ConvTranspose2d(gen_filter[2], gen_filter[2], kernel_size=2, stride=2, padding=0)
        self.up1_3 = nn.ConvTranspose2d(gen_filter[1], gen_filter[1], kernel_size=2, stride=2, padding=0)
        # 监督方式
        self.final = nn.Conv2d(gen_filter[0], 3, kernel_size=1)

    def forward(self, input_image):
        # Unet++ L1
        x0_0 = self.conv0_0(input_image)
        x1_0 = self.conv1_0(self.pool_2x(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up1_0(x1_0)], 1))
        # Unet++ L2
        x2_0 = self.conv2_0(self.pool_2x(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up2_0(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up1_1(x1_1)], 1))
        # Unet++ L3
        x3_0 = self.conv3_0(self.pool_4x(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up3_0(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up2_1(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up1_2(x1_2)], 1))
        # Unet++ L4
        x4_0 = self.conv4_0(self.pool_4x(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up4_0(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up3_1(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up2_2(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up1_3(x1_3)], 1))
        # 监督方式
        output = self.final(x0_4)
        # 最后一层激活
        output = F.sigmoid(output) - 0.5
        output = F.relu(output) + 1e-5
        return output
