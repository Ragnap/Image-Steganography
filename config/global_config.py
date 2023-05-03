# 全局常量定义文件

from easydict import EasyDict as edict

__C = edict()
cfg = __C

# 训练用参数
__C.TRAIN = edict()
__C.TRAIN.LEARNING_RATE = 0.001
# RELU中负数端的系数
__C.TRAIN.RELU_GAMA = 0.2
# 单次训练batch大小
__C.TRAIN.BATCH_SIZE = 2
# 迭代次数
__C.TRAIN.EPOCHS = 4
# 进行检查的训练数
__C.TRAIN.CHECK_EPOCHS = 2
# 生成器输入尺寸
__C.TRAIN.TRAIN_INPUT_SIZE = 256
# 训练用图片
__C.TRAIN.TRAIN_INPUT_PATH = 'dataset'
# 噪音均值
__C.TRAIN.NOISE_MEAN = 0.5
# 嵌入器中双tanh的参数
__C.TRAIN.TANH_LAMBDA = 60
# 生成器损失函数参数
__C.TRAIN.GEN_LOSS_ALPHA = 1
__C.TRAIN.GEN_LOSS_BETA = 1e-6
# 嵌入率（用于计算生成器嵌入损失）
__C.TRAIN.EMBED_RATE = 0.4
# 测试输出位置
__C.TRAIN.TRAIN_OUTPUT_PATH = 'output/train'
# 模型保存位置
__C.TRAIN.MODULE_PATH = 'output/module'


# 测试用参数
__C.TEST = edict()
