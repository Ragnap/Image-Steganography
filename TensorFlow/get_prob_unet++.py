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

global_variables = tf.global_variables()
# **************************************************************** adversary training process ***************************************************************************
image_index = range(1, NUM_IMG + 1)
seed = 0


with tf.Session() as sess:
    tf.global_variables_initializer().run()
    # ---------------loading the saved parameters------------------

    tf.train.Saver(global_variables).restore(sess, "./output/unet++/final.ckpt")
    data_x = np.zeros([BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNEL])
    count = 0
    for epoch in range(0, int(NUM_IMG/BATCH_SIZE)):

        for j in range(BATCH_SIZE):
            imc = imageio.imread('input.jpg')
            data_x[j, :, :, :] = imc


        data_noise = np.random.rand(Embeding_prob_shape[0], Embeding_prob_shape[1], Embeding_prob_shape[2], Embeding_prob_shape[3])
        prob = sess.run(Embeding_prob, feed_dict={cover: data_x, noise: data_noise, is_training: True})

        sio.savemat(prob_path + '/' + str(count) + '.mat', {'prob': prob_})
        print('processing the %d image' %count)