import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from config import global_config
CFG = global_config.cfg

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# 高通滤波器
class HPFNode(nn.Module):
    def __init__(self, image_channel=3):
        super(HPFNode, self).__init__()
        HPF = np.array([[[0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0],
                         [0, 0, -1, 1, 0],
                         [0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0]],
                        [[0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0],
                         [0, 0, -1, 0, 0],
                         [0, 0, 1, 0, 0],
                         [0, 0, 0, 0, 0]],
                        [[0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0],
                         [0, 1, -2, 1, 0],
                         [0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0]],
                        [[0, 0, 0, 0, 0],
                         [0, 0, 1, 0, 0],
                         [0, 0, -2, 0, 0],
                         [0, 0, 1, 0, 0],
                         [0, 0, 0, 0, 0]],
                        [[0, 0, 0, 0, 0],
                         [0, -1, 2, -1, 0],
                         [0, 2, -4, 2, 0],
                         [0, -1, 2, -1, 0],
                         [0, 0, 0, 0, 0]],
                        [[-1, 2, -2, 2, -1],
                         [2, -6, 8, -6, 2],
                         [-2, 8, -12, 8, -2],
                         [2, -6, 8, -6, 2],
                         [-1, 2, -2, 2, -1]]],
                       dtype=np.float32)
        self.HPF = torch.from_numpy(HPF).to(device)
        self.image_channel = image_channel

    def forward(self, input_image):
        output = []
        for i in range(6):
            kernel = self.HPF[i, :, :].unsqueeze(0).unsqueeze(0)
            kernel = kernel.repeat(1, self.image_channel, 1, 1)
            output_channel = F.conv2d(input_image, kernel, stride=1, padding=2)
            output.append(output_channel)
        output = torch.cat(output, dim=1)
        return output


class Discriminator(nn.Module):
    def __init__(self, image_size, image_channels=3):
        super(Discriminator, self).__init__()
        # 判别器特征层数量
        dis_filter = [8, 16, 32, 64, 128]
        # 高通滤波
        self.HPF = HPFNode()
        # Group 1
        self.conv_1 = nn.Conv2d(6, dis_filter[0], kernel_size=3, padding=1)
        self.bn_1 = nn.BatchNorm2d(dis_filter[0])
        self.pool_1 = nn.AvgPool2d(kernel_size=5, padding=2, stride=2)
        # Group 2
        self.conv_2 = nn.Conv2d(dis_filter[0], dis_filter[1], kernel_size=3, padding=1)
        self.bn_2 = nn.BatchNorm2d(dis_filter[1])
        self.pool_2 = nn.AvgPool2d(kernel_size=5, padding=2, stride=2)
        # Group 3
        self.conv_3 = nn.Conv2d(dis_filter[1], dis_filter[2], kernel_size=3, padding=1)
        self.bn_3 = nn.BatchNorm2d(dis_filter[2])
        self.pool_3 = nn.AvgPool2d(kernel_size=5, padding=2, stride=2)
        # Group 4
        self.conv_4 = nn.Conv2d(dis_filter[2], dis_filter[3], kernel_size=3, padding=1)
        self.bn_4 = nn.BatchNorm2d(dis_filter[3])
        self.pool_4 = nn.AvgPool2d(kernel_size=5, padding=2, stride=2)
        # Group 5
        self.conv_5 = nn.Conv2d(dis_filter[3], dis_filter[4], kernel_size=3, padding=1)
        self.bn_5 = nn.BatchNorm2d(dis_filter[4])
        self.pool_5 = torch.nn.AdaptiveAvgPool2d((1, 1))
        # Group 6 FullyConnect
        self.linear = nn.Linear(in_features=dis_filter[4], out_features=2, bias=True)

    def forward(self, input_image):
        x0 = self.HPF(input_image)
        # Group 1
        conv1 = self.conv_1(x0)
        bn1 = self.bn_1(torch.abs(conv1))
        pool1 = self.pool_1(F.tanh(bn1))
        # Group 2
        conv2 = self.conv_2(pool1)
        bn2 = self.bn_2(conv2)
        pool2 = self.pool_2(F.tanh(bn2))
        # Group 3
        conv3 = self.conv_3(pool2)
        bn3 = self.bn_3(conv3)
        pool3 = self.pool_3(F.relu(bn3))
        # Group 4
        conv4 = self.conv_4(pool3)
        bn4 = self.bn_4(conv4)
        pool4 = self.pool_4(F.relu(bn4))
        # Group 5
        conv5 = self.conv_5(pool4)
        bn5 = self.bn_5(conv5)
        pool5 = self.pool_5(F.relu(bn5))
        # Group 6 FullyConnect
        prob = pool5.squeeze()
        prob = self.linear(prob)
        prob = F.softmax(prob, dim=1)
        return prob
