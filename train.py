import os.path

import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image

import torch.nn.functional as F

from model.generator_unetpp import Generator
from model.discriminator import Discriminator
from model.embedder_double_tanh import embed

from logger import Logger

from config import global_config
CFG = global_config.cfg

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 判别器损失计算
def dis_loss_fn(x, label):
    return F.cross_entropy(x, label)


# 生成器损失计算
def gen_loss_fn(x, label, embedding_p):
    # 计算分类损失
    classify_loss = F.cross_entropy(x, label)

    # 计算嵌入损失
    # 正修改容量
    p_change_pos = embedding_p / 2.0
    # 负修改容量
    p_change_neg = embedding_p / 2.0
    # 不修改容量
    p_change_none = 1 - embedding_p
    # 有效载荷
    payload = torch.sum(- p_change_pos * torch.log2(p_change_pos) - p_change_neg * torch.log2(p_change_neg) - p_change_none * torch.log2(p_change_none), dim=[2, 3])

    embed_loss = torch.mean(torch.pow(payload - CFG.TRAIN.TRAIN_INPUT_SIZE * CFG.TRAIN.TRAIN_INPUT_SIZE * CFG.TRAIN.EMBED_RATE, 2))

    return CFG.TRAIN.GEN_LOSS_ALPHA * classify_loss + CFG.TRAIN.GEN_LOSS_BETA * embed_loss


# 读入数据集
def load_data():
    transform = transforms.Compose([
        transforms.Resize([CFG.TRAIN.TRAIN_INPUT_SIZE, CFG.TRAIN.TRAIN_INPUT_SIZE]),  # 强制改变大小为TRAIN_INPUT_SIZE
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.ToTensor(),
        transforms.Normalize(mean=[0, 0, 0], std=[1.0/255, 1.0/255, 1.0/255]),
    ])
    dataset = torchvision.datasets.ImageFolder(root=CFG.TRAIN.TRAIN_INPUT_PATH, transform=transform)
    dataloader = DataLoader(dataset, batch_size=CFG.TRAIN.BATCH_SIZE, shuffle=True, drop_last=True, num_workers=0)
    return dataloader, dataset[0][0]


# 输出单张图像的嵌入概率图与修改图像
def output_check_image(generator, check_image, epoch_time):
    path = f'{CFG.TRAIN.TRAIN_OUTPUT_PATH}/'
    if not os.path.exists(path):
        os.makedirs(path)
    check_image = torch.unsqueeze(check_image, dim=0)
    prob_image = generator(check_image)
    embed_image = embed(prob_image, check_image, torch.rand(check_image.size()).to(device) * CFG.TRAIN.NOISE_MEAN)
    output_image = torch.cat([prob_image, embed_image / 255], dim=0)
    save_image(output_image, f'{path}{epoch_time}.png', normalize=True)


# 主函数
if __name__ == '__main__':
    # 初始化日志
    logger = Logger(CFG.TRAIN.EPOCHS)

    # 初始化
    generator = Generator(CFG.TRAIN.TRAIN_INPUT_SIZE).to(device)
    discriminator = Discriminator(CFG.TRAIN.TRAIN_INPUT_SIZE).to(device)
    # Adam优化器
    gen_optimizers = torch.optim.Adam(discriminator.parameters(), lr=CFG.TRAIN.LEARNING_RATE)
    dis_optimizers = torch.optim.Adam(generator.parameters(), lr=CFG.TRAIN.LEARNING_RATE)
    # 载入训练集和测试图片
    image_dataset, check_image = load_data()
    check_image = check_image.to(device)
    # 设置标签
    real_labels = torch.zeros(CFG.TRAIN.BATCH_SIZE, 2).to(device)
    real_labels[:] = torch.tensor([0, 1])
    fake_labels = torch.zeros(CFG.TRAIN.BATCH_SIZE, 2).to(device)
    fake_labels[:] = torch.tensor([1, 0])
    # training
    for now_epoch in range(CFG.TRAIN.EPOCHS + 1):
        dis_epoch_loss = 0.0
        gen_epoch_loss = 0.0
        for i, (train_images, _) in enumerate(image_dataset):
            train_images = train_images.to(device)
            # 生成噪音
            noise = torch.rand(train_images.size()).to(device) * CFG.TRAIN.NOISE_MEAN

            # 训练判别器
            dis_optimizers.zero_grad()
            # 判别器在真实图像上面的损失
            real_images = train_images
            real_output = discriminator(real_images)
            dis_real_loss = dis_loss_fn(real_output, real_labels)
            dis_real_loss.backward()
            # 判别器在生成器生成的假图像上面的损失
            embedding_prob = generator(train_images)
            fake_images = embed(embedding_prob, train_images, noise)
            fake_output = discriminator(fake_images.detach())
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
                dis_epoch_loss += dis_loss.item()
                gen_epoch_loss += gen_loss.item()

        with torch.no_grad():
            dis_epoch_loss /= len(image_dataset)
            gen_epoch_loss /= len(image_dataset)

        # 每隔一段迭代进行一次检查
        if (now_epoch % CFG.TRAIN.CHECK_EPOCHS) == 0:
            output_check_image(generator, check_image, now_epoch)

        # 输出到日志
        logger.log_loss(now_epoch, gen_epoch_loss, dis_epoch_loss)

    # checking
    output_check_image(generator, check_image, CFG.TRAIN.EPOCHS)

    # 输出损失函数变化图像
    logger.output_loss_change_figure()

    # 保存模型
    path = f'{CFG.TRAIN.MODULE_PATH}/'
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save(generator.state_dict(), f'{path}/generator.pth')
