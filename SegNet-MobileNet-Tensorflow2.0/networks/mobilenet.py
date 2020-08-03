import tensorflow as tf
from tensorflow.keras import models, layers, activations, Input

# 对应的输入输出形状：batch_size, height, width, channels
IMAGE_ORDERING = 'channels_last'

def relu6(x):
    return activations.relu(x, max_value=6)

def conv_block(inputs, filters, alpha, kernel=(3,3), strides=(1,1)):

    channel_axis = 1 if IMAGE_ORDERING == 'channels_first' else -1
    filters = int(filters * alpha)  # juanjihe

    x = layers.ZeroPadding2D(padding=(1, 1), data_format=IMAGE_ORDERING, name='conv1_pad')(inputs)
    x = layers.Conv2D(filters, kernel, data_format=IMAGE_ORDERING, padding='valid', use_bias=False, strides=strides, name='conv1')(x)
    x = layers.BatchNormalization(axis=channel_axis, name='conv1_bn')(x)
    y = layers.Activation(relu6, name='conv1_relu')(x)

    return y

def depthwise_conv_block(inputs, pointwise_conv_filters, alpha, depth_multiplier=1, strides=(1,1), block_id=1):

    channel_axis = 1 if IMAGE_ORDERING == 'channels_first' else -1
    pointwise_conv_filters = int(pointwise_conv_filters * alpha)

    x = layers.ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING, name='conv_pad_%d' % block_id)(inputs)
    x = layers.DepthwiseConv2D((3, 3), data_format=IMAGE_ORDERING, padding='valid', depth_multiplier=depth_multiplier, strides=strides, use_bias=False, name='conv_dw_%d' % block_id)(x)
    x = layers.BatchNormalization(axis=channel_axis, name='conv_dw_%d_bn' % block_id)(x)
    x = layers.Activation(relu6, name='conv_dw_%d_relu' % block_id)(x)

    x = layers.Conv2D(pointwise_conv_filters, (1,1), data_format=IMAGE_ORDERING, padding='same', use_bias=False, strides=(1, 1), name='conv_pw_%d' % block_id)(x)
    x = layers.BatchNormalization(axis=channel_axis, name='conv_pw_%d_bn' % block_id)(x)
    y = layers.Activation(relu6, name='conv_pw_%d_relu' % block_id)(x)

    return y

def mobilenet_encoder(input_height=224, input_width=224, pretrained='imagenet'):

    alpha = 1.0
    depth_multiplier = 1
    dropout = 1e-3

    # (416, 416, 3)
    img_input = Input(shape=(input_height, input_width, 3))

    # (208, 208, 3)
    x = conv_block(img_input, 32, alpha, strides=(2, 2))
    x = depthwise_conv_block(x, 64, alpha, depth_multiplier, block_id=1)
    f1 = x

    # (104, 104, 3)
    x = depthwise_conv_block(x, 128, alpha, depth_multiplier, strides=(2, 2), block_id=2)
    x = depthwise_conv_block(x, 128, alpha, depth_multiplier, block_id=3)
    f2 = x

    # (52, 52, 3)
    x = depthwise_conv_block(x, 256, alpha, depth_multiplier, strides=(2, 2), block_id=4)
    x = depthwise_conv_block(x, 256, alpha, depth_multiplier, block_id=5)
    f3 = x

    # (26, 26, 3)
    x = depthwise_conv_block(x, 512, alpha, depth_multiplier, strides=(2, 2), block_id=6)
    x = depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=7)
    x = depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=8)
    x = depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=9)
    x = depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=10)
    x = depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=11)
    f4 = x

    x = depthwise_conv_block(x, 1024, alpha, depth_multiplier, strides=(2, 2), block_id=12)
    x = depthwise_conv_block(x, 1024, alpha, depth_multiplier, block_id=13)
    f5 = x

    return img_input, [f1, f2, f3, f4, f5]
