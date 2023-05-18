# Source code to reproduce the results in
# J. Yang, D. Ruan, J. Huang, X. Kang and Y. Shi, "An Embedding Cost Learning Framework Using GAN," in IEEE Transactions on Information Forensics and Security, vol. 15, pp. 839-851, 2020.
# By Jianhua Yang,  yangjh48@mail2.sysu.edu.cn
# import tensorflow as tf
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
import numpy as np
import random
import matplotlib.pyplot as plt
import time
import os
import imageio.v2 as imageio
import logger
from skimage.transform import resize


# import scipy.io as sio
# from batch_norm_layer import batch_norm_layer

def batch_norm_layer(x, is_training, name=None):
    bn = tf.layers.batch_normalization(
        inputs=x,
        axis=-1,
        momentum=0.05,
        epsilon=0.00001,
        center=True,
        scale=True,
        training=is_training
    )
    return bn


# select the graphic card
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
path1 = './dataset_256'  # path of training set

# ******************************************* constant value settings ************************************************
NUM_ITERATION = 1
NUM_IMG = 60  # The number of images used to train the network
BATCH_SIZE = 10
IMAGE_SIZE = 256
NUM_CHANNEL = 3  # gray image
NUM_LABELS = 2  # binary classification
G_DIM = [32, 64, 128, 256, 512]  # number of feature maps in generator
SHAPE = [256, 128, 64, 16, 4]
STRIDE = 2
KENEL_SIZE = 3
DKENEL_SIZE = 5
PAYLOAD = 0.4  # Target embedding payload
PAD_SIZE = int((KENEL_SIZE - 1) / 2)
Initial_learning_rate = 0.0001
Adam_beta = 0.5
TANH_LAMBDA = 60  # To balance the embedding simulate and avoid gradient vanish problem

cover = tf.placeholder(tf.float32, shape=[BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNEL])
is_training = tf.placeholder(tf.bool, name='is_training')  # True for training, false for test


# ********************************************* definition of the generator *********************************************************
def lrelu(x, alpha):
    return tf.nn.relu(x) - alpha * tf.nn.relu(-x)


# L1
with tf.variable_scope("Gen_0_0") as scope:
    input_data = cover
    input_channel = NUM_CHANNEL
    kernel_0_0_G = tf.Variable(tf.truncated_normal([KENEL_SIZE, KENEL_SIZE, input_channel, G_DIM[0]], stddev=0.02),
                               name="kernel_0_0_G")
    conv_0_0_G = tf.nn.conv2d(input_data, kernel_0_0_G, [1, 1, 1, 1], padding='SAME', name="conv_0_0_G")
    bn_0_0_G = lrelu(batch_norm_layer(conv_0_0_G, is_training, 'bn_0_0_G'), 0.2)
    # down
    pool_0_0_G = tf.nn.max_pool(conv_0_0_G, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name="pool_0_0_G")
    bn_0_0_down_G = lrelu(batch_norm_layer(pool_0_0_G, is_training, 'bn_0_0_down_G'), 0.2)

    # feature map shape: 128*128*32

with tf.variable_scope("Gen_1_0") as scope:
    input_data = bn_0_0_down_G
    input_channel = G_DIM[0]
    kernel_1_0_G = tf.Variable(tf.truncated_normal([KENEL_SIZE, KENEL_SIZE, input_channel, G_DIM[1]], stddev=0.02),
                               name="kernel_1_0_G")
    conv_1_0_G = tf.nn.conv2d(input_data, kernel_1_0_G, [1, 1, 1, 1], padding='SAME', name="conv_1_0_G")
    bn_1_0_G = lrelu(batch_norm_layer(conv_1_0_G, is_training, 'bn_1_0_G'), 0.2)
    # down
    pool_1_0_G = tf.nn.max_pool(conv_1_0_G, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name="pool_1_0_G")
    bn_1_0_down_G = lrelu(batch_norm_layer(pool_1_0_G, is_training, 'bn_1_0_down_G'), 0.2)
    # up
    out_shape = [BATCH_SIZE, SHAPE[0], SHAPE[0], G_DIM[1]]
    kernel_1_0_up_G = tf.Variable(tf.truncated_normal([KENEL_SIZE, KENEL_SIZE, G_DIM[1], G_DIM[1]], stddev=0.02),
                                  name="kernel_1_0_up_G")
    up_1_0_G = tf.nn.conv2d_transpose(conv_1_0_G, kernel_1_0_up_G, out_shape, [1, 2, 2, 1], name="up_1_0_G")
    bn_1_0_up_G = lrelu(batch_norm_layer(up_1_0_G, is_training, 'bn_1_0_up_G'), 0.2)

with tf.variable_scope("Gen_0_1") as scope:
    input_data = tf.concat([bn_0_0_G, bn_1_0_up_G], 3)
    input_channel = G_DIM[0] + G_DIM[1]
    kernel_0_1_G = tf.Variable(tf.truncated_normal([KENEL_SIZE, KENEL_SIZE, input_channel, G_DIM[0]], stddev=0.02),
                               name="kernel_0_1_G")
    conv_0_1_G = tf.nn.conv2d(input_data, kernel_0_1_G, [1, 1, 1, 1], padding='SAME', name="conv_0_1_G")
    bn_0_1_G = lrelu(batch_norm_layer(conv_0_1_G, is_training, 'bn_0_1_G'), 0.2)

# L2
with tf.variable_scope("Gen_2_0") as scope:
    input_data = bn_1_0_down_G
    input_channel = G_DIM[1]
    kernel_2_0_G = tf.Variable(tf.truncated_normal([KENEL_SIZE, KENEL_SIZE, input_channel, G_DIM[2]], stddev=0.02),
                               name="kernel_2_0_G")
    conv_2_0_G = tf.nn.conv2d(input_data, kernel_2_0_G, [1, 1, 1, 1], padding='SAME', name="conv_2_0_G")
    bn_2_0_G = lrelu(batch_norm_layer(conv_2_0_G, is_training, 'bn_2_0_G'), 0.2)
    # down
    pool_2_0_G = tf.nn.max_pool(conv_2_0_G, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='SAME', name="pool_2_0_G")
    bn_2_0_down_G = lrelu(batch_norm_layer(pool_2_0_G, is_training, 'bn_2_0_down_G'), 0.2)
    # up
    out_shape = [BATCH_SIZE, SHAPE[1], SHAPE[1], G_DIM[2]]
    kernel_2_0_up_G = tf.Variable(tf.truncated_normal([KENEL_SIZE, KENEL_SIZE, G_DIM[2], G_DIM[2]], stddev=0.02),
                                  name="kernel_2_0_up_G")
    up_2_0_G = tf.nn.conv2d_transpose(conv_2_0_G, kernel_2_0_up_G, out_shape, [1, 2, 2, 1], name="up_2_0_G")
    bn_2_0_up_G = lrelu(batch_norm_layer(up_2_0_G, is_training, 'bn_2_0_up_G'), 0.2)

with tf.variable_scope("Gen_1_1") as scope:
    input_data = tf.concat([bn_1_0_G, bn_2_0_up_G], 3)  # 增加通道数
    input_channel = G_DIM[1] + G_DIM[2]
    kernel_1_1_G = tf.Variable(tf.truncated_normal([KENEL_SIZE, KENEL_SIZE, input_channel, G_DIM[1]], stddev=0.02),
                               name="kernel_1_1_G")
    conv_1_1_G = tf.nn.conv2d(input_data, kernel_1_1_G, [1, 1, 1, 1], padding='SAME', name="conv_1_1_G")
    bn_1_1_G = lrelu(batch_norm_layer(conv_1_1_G, is_training, 'bn_1_1_G'), 0.2)
    # up
    out_shape = [BATCH_SIZE, SHAPE[0], SHAPE[0], G_DIM[1]]
    kernel_1_1_up_G = tf.Variable(tf.truncated_normal([KENEL_SIZE, KENEL_SIZE, G_DIM[1], G_DIM[1]], stddev=0.02),
                                  name="kernel_1_1_up_G")
    up_1_1_G = tf.nn.conv2d_transpose(conv_1_1_G, kernel_1_1_up_G, out_shape, [1, 2, 2, 1], name="up_1_1_G")
    bn_1_1_up_G = lrelu(batch_norm_layer(up_1_1_G, is_training, 'bn_1_1_up_G'), 0.2)

with tf.variable_scope("Gen_0_2") as scope:
    input_data = tf.concat([bn_0_0_G, bn_0_1_G, bn_1_1_up_G], 3)
    input_channel = 2 * G_DIM[0] + G_DIM[1]
    kernel_0_2_G = tf.Variable(tf.truncated_normal([KENEL_SIZE, KENEL_SIZE, input_channel, G_DIM[0]], stddev=0.02),
                               name="kernel_0_2_G")
    conv_0_2_G = tf.nn.conv2d(input_data, kernel_0_2_G, [1, 1, 1, 1], padding='SAME', name="conv_0_2_G")
    bn_0_2_G = lrelu(batch_norm_layer(conv_0_2_G, is_training, 'bn_0_2_G'), 0.2)

# L3
with tf.variable_scope("Gen_3_0") as scope:
    input_data = bn_2_0_down_G
    input_channel = G_DIM[2]
    kernel_3_0_G = tf.Variable(tf.truncated_normal([KENEL_SIZE, KENEL_SIZE, input_channel, G_DIM[3]], stddev=0.02),
                               name="kernel_3_0_G")
    conv_3_0_G = tf.nn.conv2d(input_data, kernel_3_0_G, [1, 1, 1, 1], padding='SAME', name="conv_3_0_G")
    bn_3_0_G = lrelu(batch_norm_layer(conv_3_0_G, is_training, 'bn_3_0_G'), 0.2)
    # down
    pool_3_0_G = tf.nn.max_pool(conv_3_0_G, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='SAME', name="pool_3_0_G")
    bn_3_0_down_G = lrelu(batch_norm_layer(pool_3_0_G, is_training, 'bn_3_0_down_G'), 0.2)
    # up
    out_shape = [BATCH_SIZE, SHAPE[2], SHAPE[2], G_DIM[3]]
    kernel_3_0_up_G = tf.Variable(tf.truncated_normal([KENEL_SIZE, KENEL_SIZE, G_DIM[3], G_DIM[3]], stddev=0.02),
                                  name="kernel_3_0_up_G")
    up_3_0_G = tf.nn.conv2d_transpose(conv_3_0_G, kernel_3_0_up_G, out_shape, [1, 4, 4, 1], name="up_3_0_G")
    bn_3_0_up_G = lrelu(batch_norm_layer(up_3_0_G, is_training, 'bn_3_0_up_G'), 0.2)

with tf.variable_scope("Gen_2_1") as scope:
    input_data = tf.concat([bn_2_0_G, bn_3_0_up_G], 3)
    input_channel = G_DIM[2] + G_DIM[3]
    kernel_2_1_G = tf.Variable(tf.truncated_normal([KENEL_SIZE, KENEL_SIZE, input_channel, G_DIM[2]], stddev=0.02),
                               name="kernel_2_1_G")
    conv_2_1_G = tf.nn.conv2d(input_data, kernel_2_1_G, [1, 1, 1, 1], padding='SAME', name="conv_2_1_G")
    bn_2_1_G = lrelu(batch_norm_layer(conv_2_1_G, is_training, 'bn_2_1_G'), 0.2)
    # up
    out_shape = [BATCH_SIZE, SHAPE[1], SHAPE[1], G_DIM[2]]
    kernel_2_1_up_G = tf.Variable(tf.truncated_normal([KENEL_SIZE, KENEL_SIZE, G_DIM[2], G_DIM[2]], stddev=0.02),
                                  name="kernel_2_1_up_G")
    up_2_1_G = tf.nn.conv2d_transpose(conv_2_1_G, kernel_2_1_up_G, out_shape, [1, 2, 2, 1], name="up_2_1_G")
    bn_2_1_up_G = lrelu(batch_norm_layer(up_2_1_G, is_training, 'bn_2_1_up_G'), 0.2)

with tf.variable_scope("Gen_1_2") as scope:
    input_data = tf.concat([bn_1_0_G, bn_1_1_G, bn_2_1_up_G], 3)
    input_channel = 2 * G_DIM[1] + G_DIM[2]
    kernel_1_2_G = tf.Variable(tf.truncated_normal([KENEL_SIZE, KENEL_SIZE, input_channel, G_DIM[1]], stddev=0.02),
                               name="kernel_1_2_G")
    conv_1_2_G = tf.nn.conv2d(input_data, kernel_1_2_G, [1, 1, 1, 1], padding='SAME', name="conv_1_2_G")
    bn_1_2_G = lrelu(batch_norm_layer(conv_1_2_G, is_training, 'bn_1_2_G'), 0.2)
    # up
    out_shape = [BATCH_SIZE, SHAPE[0], SHAPE[0], G_DIM[1]]
    kernel_1_2_up_G = tf.Variable(tf.truncated_normal([KENEL_SIZE, KENEL_SIZE, G_DIM[1], G_DIM[1]], stddev=0.02),
                                  name="kernel_1_2_up_G")
    up_1_2_G = tf.nn.conv2d_transpose(conv_1_2_G, kernel_1_2_up_G, out_shape, [1, 2, 2, 1], name="up_1_2_G")
    bn_1_2_up_G = lrelu(batch_norm_layer(up_1_2_G, is_training, 'bn_1_2_up_G'), 0.2)

with tf.variable_scope("Gen_0_3") as scope:
    input_data = tf.concat([bn_0_0_G, bn_0_1_G, bn_0_2_G, bn_1_2_up_G], 3)
    input_channel = 3 * G_DIM[0] + G_DIM[1]
    kernel_0_3_G = tf.Variable(tf.truncated_normal([KENEL_SIZE, KENEL_SIZE, input_channel, G_DIM[0]], stddev=0.02),
                               name="kernel_0_3_G")
    conv_0_3_G = tf.nn.conv2d(input_data, kernel_0_3_G, [1, 1, 1, 1], padding='SAME', name="conv_0_3_G")
    bn_0_3_G = lrelu(batch_norm_layer(conv_0_3_G, is_training, 'bn_0_3_G'), 0.2)

# L4
with tf.variable_scope("Gen_4_0") as scope:
    input_data = bn_3_0_down_G
    input_channel = G_DIM[3]
    kernel_4_0_G = tf.Variable(tf.truncated_normal([KENEL_SIZE, KENEL_SIZE, input_channel, G_DIM[4]], stddev=0.02),
                               name="kernel_4_0_G")
    conv_4_0_G = tf.nn.conv2d(input_data, kernel_4_0_G, [1, 1, 1, 1], padding='SAME', name="conv_4_0_G")
    bn_4_0_G = lrelu(batch_norm_layer(conv_4_0_G, is_training, 'bn_4_0_G'), 0.2)
    # up
    out_shape = [BATCH_SIZE, SHAPE[3], SHAPE[3], G_DIM[4]]
    kernel_4_0_up_G = tf.Variable(tf.truncated_normal([KENEL_SIZE, KENEL_SIZE, G_DIM[4], G_DIM[4]], stddev=0.02),
                                  name="kernel_4_0_up_G")
    up_4_0_G = tf.nn.conv2d_transpose(conv_4_0_G, kernel_4_0_up_G, out_shape, [1, 4, 4, 1], name="up_4_0_G")
    bn_4_0_up_G = lrelu(batch_norm_layer(up_4_0_G, is_training, 'bn_4_0_up_G'), 0.2)

with tf.variable_scope("Gen_3_1") as scope:
    input_data = tf.concat([bn_3_0_G, bn_4_0_up_G], 3)
    input_channel = G_DIM[3] + G_DIM[4]
    kernel_3_1_G = tf.Variable(tf.truncated_normal([KENEL_SIZE, KENEL_SIZE, input_channel, G_DIM[3]], stddev=0.02),
                               name="kernel_3_1_G")
    conv_3_1_G = tf.nn.conv2d(input_data, kernel_3_1_G, [1, 1, 1, 1], padding='SAME', name="conv_3_1_G")
    # up
    out_shape = [BATCH_SIZE, SHAPE[2], SHAPE[2], G_DIM[3]]
    kernel_3_1_up_G = tf.Variable(tf.truncated_normal([KENEL_SIZE, KENEL_SIZE, G_DIM[3], G_DIM[3]], stddev=0.02),
                                  name="kernel_3_1_up_G")
    up_3_1_G = tf.nn.conv2d_transpose(conv_3_1_G, kernel_3_1_up_G, out_shape, [1, 4, 4, 1], name="up_3_1_G")
    bn_3_1_up_G = lrelu(batch_norm_layer(up_3_1_G, is_training, 'bn_3_1_up_G'), 0.2)

with tf.variable_scope("Gen_2_2") as scope:
    input_data = tf.concat([bn_2_0_G, bn_2_1_G, bn_3_1_up_G], 3)
    input_channel = 2 * G_DIM[2] + G_DIM[3]
    kernel_2_2_G = tf.Variable(tf.truncated_normal([KENEL_SIZE, KENEL_SIZE, input_channel, G_DIM[2]], stddev=0.02),
                               name="kernel_2_2_G")
    conv_2_2_G = tf.nn.conv2d(input_data, kernel_2_2_G, [1, 1, 1, 1], padding='SAME', name="conv_2_2_G")
    # up
    out_shape = [BATCH_SIZE, SHAPE[1], SHAPE[1], G_DIM[2]]
    kernel_2_2_up_G = tf.Variable(tf.truncated_normal([KENEL_SIZE, KENEL_SIZE, G_DIM[2], G_DIM[2]], stddev=0.02),
                                  name="kernel_2_2_up_G")
    up_2_2_G = tf.nn.conv2d_transpose(conv_2_2_G, kernel_2_2_up_G, out_shape, [1, 2, 2, 1], name="up_2_2_G")
    bn_2_2_up_G = lrelu(batch_norm_layer(up_2_2_G, is_training, 'bn_2_2_up_G'), 0.2)

with tf.variable_scope("Gen_1_3") as scope:
    input_data = tf.concat([bn_1_0_G, bn_1_1_G, bn_1_2_G, bn_2_2_up_G], 3)
    input_channel = 3 * G_DIM[1] + G_DIM[2]
    kernel_1_3_G = tf.Variable(tf.truncated_normal([KENEL_SIZE, KENEL_SIZE, input_channel, G_DIM[1]], stddev=0.02),
                               name="kernel_1_3_G")
    conv_1_3_G = tf.nn.conv2d(input_data, kernel_1_3_G, [1, 1, 1, 1], padding='SAME', name="conv_1_3_G")
    # up
    out_shape = [BATCH_SIZE, SHAPE[0], SHAPE[0], G_DIM[1]]
    kernel_1_3_up_G = tf.Variable(tf.truncated_normal([KENEL_SIZE, KENEL_SIZE, G_DIM[1], G_DIM[1]], stddev=0.02),
                                  name="kernel_1_3_up_G")
    up_1_3_G = tf.nn.conv2d_transpose(conv_1_3_G, kernel_1_3_up_G, out_shape, [1, 2, 2, 1], name="up_1_3_G")
    bn_1_3_up_G = lrelu(batch_norm_layer(up_1_3_G, is_training, 'bn_1_3_up_G'), 0.2)

with tf.variable_scope("Gen_0_4") as scope:
    input_data = tf.concat([bn_0_0_G, bn_0_1_G, bn_0_2_G, bn_0_3_G, bn_1_3_up_G], 3)
    input_channel = 4 * G_DIM[0] + G_DIM[1]
    kernel_0_4_G = tf.Variable(tf.truncated_normal([KENEL_SIZE, KENEL_SIZE, input_channel, G_DIM[0]], stddev=0.02),
                               name="kernel_0_4_G")
    conv_0_4_G = tf.nn.conv2d(input_data, kernel_0_4_G, [1, 1, 1, 1], padding='SAME', name="conv_0_4_G")
    bn_0_4_G = lrelu(batch_norm_layer(conv_0_4_G, is_training, 'bn_0_4_G'), 0.2)

with tf.variable_scope("Gen_f") as scope:
    input_data = bn_0_4_G
    input_channel = G_DIM[0]
    kernel_f_G = tf.Variable(tf.truncated_normal([KENEL_SIZE, KENEL_SIZE, input_channel, NUM_CHANNEL], stddev=0.02),
                             name="kernel_f_G")
    conv_f_G = tf.nn.conv2d(input_data, kernel_f_G, [1, 1, 1, 1], padding='SAME', name="conv_f_G")
    bn_f_G = lrelu(batch_norm_layer(conv_f_G, is_training, 'bn_f_G'), 0.2)

Embeding_prob = tf.nn.relu(tf.nn.sigmoid(bn_f_G) - 0.5)
Embeding_prob_shape = Embeding_prob.get_shape().as_list()

# ***************************************************  double-tanh function for embedding simulation ***************************************************
noise = tf.placeholder(tf.float32, Embeding_prob_shape)  # noise holder
modification = -0.5 * tf.nn.tanh((tf.subtract(Embeding_prob, 2 * noise)) * TANH_LAMBDA) + 0.5 * tf.nn.tanh(
    (tf.subtract(Embeding_prob, tf.subtract(2.0, 2 * noise))) * TANH_LAMBDA)
stego = cover + modification

# *************************************************** definition of the discriminator **************************************************************
Img = tf.concat([cover, stego], 0)
y_array = np.zeros([BATCH_SIZE * 2, NUM_LABELS], dtype=np.float32)
for i in range(0, BATCH_SIZE):
    y_array[i, 1] = 1
for i in range(BATCH_SIZE, BATCH_SIZE * 2):
    y_array[i, 0] = 1
y = tf.constant(y_array)

Img_label = tf.constant(y_array)

# *********************** high pass filters ***********************
HPF = np.zeros([5, 5, NUM_CHANNEL, 6], dtype=np.float32)
HPF[:, :, 0, 0] = np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, -1, 1, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
                           dtype=np.float32)
HPF[:, :, 0, 1] = np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, -1, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0]],
                           dtype=np.float32)
HPF[:, :, 0, 2] = np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 1, -2, 1, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
                           dtype=np.float32)
HPF[:, :, 0, 3] = np.array([[0, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, -2, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0]],
                           dtype=np.float32)
HPF[:, :, 0, 4] = np.array([[0, 0, 0, 0, 0], [0, -1, 2, -1, 0], [0, 2, -4, 2, 0], [0, -1, 2, -1, 0], [0, 0, 0, 0, 0]],
                           dtype=np.float32)
HPF[:, :, 0, 5] = np.array(
    [[-1, 2, -2, 2, -1], [2, -6, 8, -6, 2], [-2, 8, -12, 8, -2], [2, -6, 8, -6, 2], [-1, 2, -2, 2, -1]],
    dtype=np.float32)

skernel0 = tf.Variable(HPF, name="skernel0")
sconv0 = tf.nn.conv2d(Img, skernel0, [1, 1, 1, 1], 'SAME', name="sconv0")

with tf.variable_scope("Group1") as scope:
    skernel1 = tf.Variable(tf.random_normal([5, 5, 6, 8], mean=0.0, stddev=0.01), name="skernel1")
    sconv1 = tf.nn.conv2d(sconv0, skernel1, [1, 1, 1, 1], padding='SAME', name="sconv1")
    sabs1 = tf.abs(sconv1, name="sabs1")
    sbn1 = batch_norm_layer(sabs1, is_training, 'sbn1')
    stanh1 = tf.nn.tanh(sbn1, name="stanh1")
    spool1 = tf.nn.avg_pool(stanh1, ksize=[1, 5, 5, 1], strides=[1, 2, 2, 1], padding='SAME', name="spool1")

with tf.variable_scope("Group2") as scope:
    skernel2 = tf.Variable(tf.random_normal([5, 5, 8, 16], mean=0.0, stddev=0.01), name="skernel2")
    sconv2 = tf.nn.conv2d(spool1, skernel2, [1, 1, 1, 1], padding="SAME", name="sconv2")
    sbn2 = batch_norm_layer(sconv2, is_training, 'sbn2')
    stanh2 = tf.nn.tanh(sbn2, name="stanh2")
    spool2 = tf.nn.avg_pool(stanh2, ksize=[1, 5, 5, 1], strides=[1, 2, 2, 1], padding='SAME', name="spool2")

with tf.variable_scope("Group3") as scope:
    skernel3 = tf.Variable(tf.random_normal([1, 1, 16, 32], mean=0.0, stddev=0.01), name="skernel3")
    sconv3 = tf.nn.conv2d(spool2, skernel3, [1, 1, 1, 1], padding="SAME", name="sconv3")
    sbn3 = batch_norm_layer(sconv3, is_training, 'sbn3')
    srelu3 = tf.nn.relu(sbn3, name="sbn3")
    spool3 = tf.nn.avg_pool(srelu3, ksize=[1, 5, 5, 1], strides=[1, 2, 2, 1], padding="SAME",
                            name="spool3")  # [input,height,width,ouput]

with tf.variable_scope("Group4") as scope:
    skernel4 = tf.Variable(tf.random_normal([1, 1, 32, 64], mean=0.0, stddev=0.01), name="skernel4")
    sconv4 = tf.nn.conv2d(spool3, skernel4, [1, 1, 1, 1], padding="SAME", name="sconv4")
    sbn4 = batch_norm_layer(sconv4, is_training, 'sbn4')
    srelu4 = tf.nn.relu(sbn4, name="srelu4")
    spool4 = tf.nn.avg_pool(srelu4, ksize=[1, 5, 5, 1], strides=[1, 2, 2, 1], padding="SAME",
                            name="spool4")  # [input,height,width,ouput]

with tf.variable_scope("Group5") as scope:
    skernel5 = tf.Variable(tf.random_normal([1, 1, 64, 128], mean=0.0, stddev=0.01), name="skernel5")
    sconv5 = tf.nn.conv2d(spool4, skernel5, [1, 1, 1, 1], padding="SAME", name="sconv5")
    sbn5 = batch_norm_layer(sconv5, is_training, 'sbn5')
    srelu5 = tf.nn.relu(sbn5, name="srelu5")
    spool5 = tf.nn.avg_pool(srelu5, ksize=[1, 16, 16, 1], strides=[1, 1, 1, 1], padding="VALID",
                            name="spool5")  # [input,height,width,ouput]

with tf.variable_scope('Group6') as scope:
    spool_shape = spool5.get_shape().as_list()
    spool_reshape = tf.reshape(spool5, [spool_shape[0], spool_shape[1] * spool_shape[2] * spool_shape[3]])
    sweights = tf.Variable(tf.random_normal([128, 2], mean=0.0, stddev=0.01), name="sweights")
    sbias = tf.Variable(tf.random_normal([2], mean=0.0, stddev=0.01), name="sbias")
    D_y = tf.matmul(spool_reshape, sweights) + sbias

correct_predictionS = tf.equal(tf.argmax(D_y, 1), tf.argmax(Img_label, 1))
accuracyD = tf.reduce_mean(tf.cast(correct_predictionS, tf.float32))
lossD = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=D_y, labels=Img_label))  # loss of D

# *******************************************************loss function ************************************************************************
gamma = 1
lambda_ent = 1e-7
proChangeP = Embeding_prob / 2.0 + 1e-5
proChangeM = Embeding_prob / 2.0 + 1e-5
proUnchange = 1 - Embeding_prob + 1e-5
entropy = tf.reduce_sum(- (proChangeP) * tf.log(proChangeP) / tf.log(2.0)
                        - (proChangeM) * tf.log(proChangeM) / tf.log(2.0)
                        - proUnchange * tf.log(proUnchange) / tf.log(2.0), reduction_indices=[1, 2, 3])
Payload_learned = tf.reduce_sum(entropy, reduction_indices=0) / IMAGE_SIZE / IMAGE_SIZE / BATCH_SIZE

Capacity = IMAGE_SIZE * IMAGE_SIZE * PAYLOAD
lossEntropy = tf.reduce_mean(tf.pow(entropy - Capacity, 2), reduction_indices=0)
# -------------------loss of the generator -------------
lossGen = gamma * (-lossD) + lambda_ent * lossEntropy
# -------------------trainable variables----------------
variables = tf.trainable_variables()
paramsG = [v for v in variables if (v.name.startswith('Gen'))]
paramsD = [v for v in variables if (v.name.startswith('Group'))]

# -------------------- optimizers ---------------------------
with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    optG = tf.train.AdamOptimizer(Initial_learning_rate).minimize(lossGen, var_list=paramsG)
    optD = tf.train.AdamOptimizer(Initial_learning_rate).minimize(lossD, var_list=paramsD)

global_variables = tf.global_variables()
# **************************************************************** adversary training process ***************************************************************************
image_index = range(1, NUM_IMG + 1)
seed = 0
logger = logger.Logger(NUM_ITERATION - 1, './output/unet++')
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    iteration_num = 0
    data_x = np.zeros([BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNEL])
    count = 0
    start_time = time.time()
    for Iter_ in range(0 + 1, NUM_ITERATION + 1):
        for j in range(BATCH_SIZE):
            count = count % NUM_IMG
            if (count == 0):
                print('----------- Epoch %d------------' % seed)
                np.random.seed(seed)
                seed = seed + 1
                temp_image_index = np.random.permutation(image_index)  # shuffle the training set every epoch
            # imc = ndimage.imread(path1 + '/60' + '%02d' %temp_image_index[count] + '.tif')  # %06d tif
            imc = imageio.imread(path1 + '/60' + '%02d' % temp_image_index[count] + '.jpg')  # %06d tif
            # imc = resize(imc, (256, 256))

            data_x[j, :, :, :] = imc
            count = count + 1

        if Iter_ % 5000 == 0:
            saver = tf.train.Saver()
            saver.save(sess, './output/unet++/model/' + '%d' % Iter_ + '.ckpt')

        if Iter_ == NUM_ITERATION:
            saver = tf.train.Saver()
            saver.save(sess, './output/unet++/final.ckpt')

        data_noise = np.random.rand(Embeding_prob_shape[0], Embeding_prob_shape[1], Embeding_prob_shape[2],
                                    Embeding_prob_shape[3])
        # update S
        sess.run(optD, feed_dict={cover: data_x, noise: data_noise, is_training: True})

        # update G
        _, lG, payload_learned, OUT, modified, accD, loD, = sess.run(
            [optG, lossGen, Payload_learned, Embeding_prob, modification,
             accuracyD, lossD],
            feed_dict={cover: data_x, noise: data_noise, is_training: True})

        spend_time = time.time() - start_time
        if Iter_ % 1 == 0:
            logger.log_loss(Iter_, lG, loD, spend_time)
            logger.log_accuracy(Iter_, accuracy=accD)
            # print('Iter %d' % Iter_ + '\tlossG=%f' % lG + '\tPayload=%f' % payload_learned + '\tlossD =%f' % loD
            #       + '\taccuracyD = %f' % accD + '\tspend time:  %f' % spend_time + 's')
        if Iter_ % 20 == 0:
            cover_data, prob_data, stego_data = sess.run([cover, Embeding_prob, stego],
                                                         feed_dict={cover: data_x, noise: data_noise,
                                                                    is_training: True})
            imageio.imwrite(f'./output/log-unet++/pic/{Iter_}cover.jpg', cover_data[0])
            imageio.imwrite(f'./output/log-unet++/pic/{Iter_}prob.jpg', prob_data[0])
            imageio.imwrite(f'./output/log-unet++/pic/{Iter_}stego.jpg', stego_data[0])
    # writer.close()

logger.output_loss_change_figure()
logger.output_acc_change_figure()

