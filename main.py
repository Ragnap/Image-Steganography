import os.path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image

# 超参数定义
# RELU中负数端的系数
RELU_GAMA = 0.2
# 单次训练batch大小
BATCH_SIZE = 2
# 迭代次数
EPOCHS = 20
# 进行检查的训练数
CHECK_EPOCHS = [EPOCHS / 4, EPOCHS / 2, EPOCHS * 3 / 4]
# 生成器输入尺寸
TRAIN_INPUT_SIZE = 256
# 训练用图片
TRAIN_INPUT_PATH = 'dataset'
# 测试输出位置
TRAIN_OUTPUT_PATH = 'output/train'
# 模型保存位置
MODULE_PATH = 'output/module'
# 噪音均值
NOISE_MEAN = 0.5
# 嵌入器中双tanh的参数
TANH_LAMBDA = 60
# 生成器损失函数参数
GEN_LOSS_ALPHA = 1
GEN_LOSS_BETA = 1e-4
# 嵌入率（用于计算生成器嵌入损失）
EMBED_RATE = 0.4

# 获取当前运行位置
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# 生成器网络节点
class GeneratorNode(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(negative_slope=RELU_GAMA)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out


# 嵌入器
def embed(prob, image, noise):
    output = -0.5 * F.tanh((prob - 2 * noise) * TANH_LAMBDA) + 0.5 * F.tanh((prob - (2.0 - 2 * noise)) * TANH_LAMBDA)
    return output + image


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


# 判别器
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
        self.pool_5 = torch.nn.AdaptiveAvgPool2d(dis_filter[4])
        # Group 6 FullyConnect
        self.linear = nn.Linear(in_features=dis_filter[4], out_features=2)


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


# 判别器损失计算
def dis_loss_fn(x, label):
    cross_entropy = torch.nn.BCELoss(reduction='sum')
    return cross_entropy(x, label)


# 生成器损失计算
def gen_loss_fn(x, label, embedding_p):
    # 计算分类损失
    cross_entropy = torch.nn.BCELoss(reduction='sum')
    classify_loss = cross_entropy(x, label)

    # 计算嵌入损失
    # 正修改容量
    p_change_pos = embedding_p / 2.0
    # 负修改容量
    p_change_neg = embedding_p / 2.0
    # 不修改容量
    p_change_none = 1 - embedding_p
    # 有效载荷
    payload = - p_change_pos * torch.log2(p_change_pos) - p_change_neg * torch.log2(p_change_neg) - p_change_none * torch.log2(p_change_none)
    embed_loss = torch.pow(payload.sum() - TRAIN_INPUT_SIZE * TRAIN_INPUT_SIZE * EMBED_RATE, 2)

    return GEN_LOSS_ALPHA * classify_loss + GEN_LOSS_BETA * embed_loss


# 读入数据集
def load_data():
    transform = transforms.Compose([
        transforms.Resize([TRAIN_INPUT_SIZE, TRAIN_INPUT_SIZE]),  # 强制改变大小为TRAIN_INPUT_SIZE
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    dataset = torchvision.datasets.ImageFolder(root=TRAIN_INPUT_PATH, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=0)
    return dataloader, dataset[0][0]


def output_origin_image(origin_image, image_id, path=''):
    path = f'{TRAIN_OUTPUT_PATH}/{path}'
    if not os.path.exists(path):
        os.makedirs(path)
    save_image(origin_image, f'{path}ori_images{image_id}.png', nrow=BATCH_SIZE, normalize=True)


def output_prob_image(generator, origin_images, image_id, path=''):
    path = f'{TRAIN_OUTPUT_PATH}/{path}'
    if not os.path.exists(path):
        os.makedirs(path)
    images = generator(origin_images.detach())
    save_image(images, f'{path}prob_images{image_id}.png', nrow=BATCH_SIZE, normalize=True)


def output_embed_image(generator, origin_images, image_id, path=''):
    path = f'{TRAIN_OUTPUT_PATH}/{path}'
    if not os.path.exists(path):
        os.makedirs(path)
    prob = generator(origin_images.detach())
    images = embed(prob, origin_images, noise)
    save_image(images, f'{path}embed_images{image_id}.png', nrow=BATCH_SIZE, normalize=True)


def output_check_image(generator, check_image, epoch_time):
    path = f'{TRAIN_OUTPUT_PATH}/'
    if not os.path.exists(path):
        os.makedirs(path)
    check_image = torch.unsqueeze(check_image, dim=0)
    prob_image = generator(check_image)
    embed_image = embed(prob_image, check_image, torch.rand(check_image.size()).to(device) * NOISE_MEAN)
    output_image = torch.cat([prob_image, embed_image], dim=0)
    save_image(output_image, f'{path}{epoch_time}.png', normalize=True)


# 主函数
if __name__ == '__main__':
    # 初始化
    generator = Generator(TRAIN_INPUT_SIZE).to(device)
    discriminator = Discriminator(TRAIN_INPUT_SIZE).to(device)
    # Adam优化器
    gen_optimizers = torch.optim.Adam(discriminator.parameters(), lr=0.0001)
    dis_optimizers = torch.optim.Adam(generator.parameters(), lr=0.0001)
    # 载入训练集和测试图片
    image_dataset, check_image = load_data()
    check_image = check_image.to(device)
    # training
    for now_epoch in range(EPOCHS + 1):
        dis_epoch_loss = 0.0
        gen_epoch_loss = 0.0
        for i, (train_images, _) in enumerate(image_dataset):
            train_images = train_images.to(device)
            # 生成噪音
            noise = torch.rand(train_images.size()).to(device) * NOISE_MEAN

            # 训练判别器
            dis_optimizers.zero_grad()
            # 判别器在真实图像上面的损失
            real_images = train_images
            real_output = discriminator(real_images.detach())
            real_labels = torch.ones_like(real_output)
            dis_real_loss = dis_loss_fn(real_output, real_labels)
            dis_real_loss.backward()
            # 判别器在生成器生成的假图像上面的损失
            embedding_prob = generator(train_images.detach())
            fake_images = embed(embedding_prob, train_images, noise)
            fake_output = discriminator(fake_images.detach())
            fake_labels = torch.zeros_like(fake_output)
            dis_fake_loss = dis_loss_fn(fake_output, fake_labels)
            dis_fake_loss.backward()
            # 判别器总损失相加
            dis_loss = dis_real_loss + dis_fake_loss
            dis_optimizers.step()

            # 训练生成器
            gen_optimizers.zero_grad()
            embedding_prob = generator(train_images)
            fake_images = embed(embedding_prob, train_images, noise)
            fake_output = discriminator(fake_images)
            gen_loss = gen_loss_fn(fake_output, real_labels, embedding_prob)
            gen_loss.backward()
            gen_optimizers.step()

            with torch.no_grad():
                dis_epoch_loss += dis_loss
                gen_epoch_loss += gen_loss

        # 每隔一段迭代进行一次检查
        if now_epoch in CHECK_EPOCHS:
            output_check_image(generator, check_image, now_epoch)
            # for image_id, (origin_images, _) in enumerate(image_dataset):
            #     origin_images = origin_images.to(device)
            #     output_origin_image(origin_images, image_id, f'{epoch}/')
            #     output_prob_image(generator, origin_images, image_id, f'{epoch}/')
            #     output_embed_image(generator, origin_images, image_id, f'{epoch}/')

        print(f"Epoch: {now_epoch:6} - loss:{gen_epoch_loss.item() + dis_epoch_loss.item():10} [gen:{gen_epoch_loss.item():10}, dis:{dis_epoch_loss.item():10}]")

    # checking
    output_check_image(generator, check_image, EPOCHS)
    # for i, (train_images, _) in enumerate(image_dataset):
    #     train_images = train_images.to(device)
    #     output_origin_image(train_images, i, f'{EPOCHS}/')
    #     output_prob_image(generator, train_images, i, f'{EPOCHS}/')
    #     output_embed_image(generator, train_images, i, f'{EPOCHS}/')
    # 保存模型
    path = f'{MODULE_PATH}/'
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save(generator.state_dict(), f'{path}/generator.pth')


