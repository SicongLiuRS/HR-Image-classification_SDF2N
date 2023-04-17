from tensorflow.keras import layers
from tensorflow.keras import backend as K
import tensorflow as tf
import math
from tensorflow.keras.layers import *
import torch

ratio = 4
def attach_attention_module(net, attention_module):
    if attention_module == 'se_block':  # SE
        net = se_block(net)
    elif attention_module == 'see_block': # SEE
        net = see_block(net)
    elif attention_module == 'sk_block': # SK
        net = sk_block(net)
    elif attention_module == 'pcm_block': # PCM
        net = pcm_block(net)
    elif attention_module == 'cbam_block':  # CBAM
        net = cbam_block(net)
    elif attention_module == 'danet_block':  # DAnet
        net = danet_block(net)
    elif attention_module == 'cf_block':  # CF
        net = danet_block(net)
    elif attention_module == 'eca_block':  # CF
        net = eca_block(net)
    else:
        raise Exception("'{}' is not supported attention module!".format(attention_module))

    return net

def cf_block(input_feature, reduction = ratio):
    channel = input_feature.shape.as_list()[-1]
    conv1 = layers.Conv2D(32, (1, 1), activation='relu', padding='same')(input_feature)
    conv2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_feature)
    conate = layers.concatenate([conv1, conv2])

    conate = layers.GlobalAveragePooling2D()(conate)
    conv3 = layers.Dense(channel // reduction, activation='relu')(conate)
    conv4 = layers.Dense(channel, activation='sigmoid')(conv3)
    conv4 = layers.Reshape((1, 1, channel))(conv4)

    return layers.Multiply()([conate, conv4])

def pcm_block(input_feature):
    feature = input_feature
    cam = input_feature
    inputs_shape = feature.get_shape().as_list()
    n, h, w, c = inputs_shape[0], inputs_shape[1], inputs_shape[2],inputs_shape[3]
    scale = layers.Lambda(lambda x: tf.image.resize_images(x, (h*w, 1),0))(cam)
    scale = layers.Reshape((h * w, c))(scale)
    f = layers.Conv2D(int(c), kernel_size=(1, 1), strides=(1, 1))(feature)
    f = layers.Lambda(lambda x: tf.image.resize_images(x, (h*w, 1),0))(f)
    f = layers.Lambda(lambda x: x/(tf.norm(x)+1e-5))(f)
    ft = layers.Lambda(lambda x: tf.image.resize_images(x, (1, h * w), 0))(f)
    f = layers.Reshape((c, h * w))(f)
    ft = layers.Reshape((h * w, c))(ft)
    f = layers.Lambda(lambda x: tf.matmul(x[0], x[1]))([ft, f])
    f = layers.Activation('relu')(f)
    x = layers.Lambda(lambda x: tf.matmul(x[0], x[1]))([f, scale])
    x = layers.Reshape((h, w, c))(x)
    return x

# def se_block(input_feature, ratio=4):
#     """Contains the implementation of Squeeze-and-Excitation(SE) block.
#     As described in https://arxiv.org/abs/1709.01507.
#     """
#
#     channel_axis = 1 if K.image_data_format() == "channels_first" else -1
#     channel = input_feature._keras_shape[channel_axis]
#
#     se_feature = layers.GlobalAveragePooling2D()(input_feature)
#     se_feature = layers.Reshape((1, 1, channel))(se_feature)
#     assert se_feature._keras_shape[1:] == (1, 1, channel)
#     se_feature = layers.Dense(channel // ratio,
#                               activation='relu',
#                               kernel_initializer='he_normal',
#                               use_bias=True,
#                               bias_initializer='zeros')(se_feature)
#     assert se_feature._keras_shape[1:] == (1, 1, channel // ratio)
#     se_feature = layers.Dense(channel,
#                               activation='sigmoid',
#                               kernel_initializer='he_normal',
#                               use_bias=True,
#                               bias_initializer='zeros')(se_feature)
#     assert se_feature._keras_shape[1:] == (1, 1, channel)
#     if K.image_data_format() == 'channels_first':
#         se_feature = layers.Permute((3, 1, 2))(se_feature)
#
#     se_feature = layers.multiply([input_feature, se_feature])
#     return se_feature

def se_block(input_feature, reduction=ratio):
    channel = input_feature.shape.as_list()[-1]
    squeeze = layers.GlobalAveragePooling2D()(input_feature)
    excitation = layers.Dense(channel // reduction, activation='relu')(squeeze)
    excitation = layers.Dense(channel, activation='sigmoid')(excitation)
    excitation = layers.Reshape((1, 1, channel))(excitation)

    return layers.Multiply()([input_feature, excitation])

def see_block(input_feature, reduction=ratio):

    channel = input_feature.shape.as_list()[-1]
    squeeze = layers.GlobalAveragePooling2D()(input_feature)
    excitation = layers.Dense(channel // reduction, activation='relu')(squeeze)
    excitation = layers.Dense(channel, activation='sigmoid')(excitation)
    excitation_1 = layers.Reshape((1, 1, channel))(excitation)
    se_feature = layers.Multiply()([input_feature, excitation_1])

    # split_feature = tf.unstack(se_feature, axis=0)
    # split_index = tf.unstack(index, axis=0)
    # output_feature_list = list()
    # for i in range(128):
    #     output_feature_list.append(tf.gather(split_feature[i], split_index[i], axis=-1))
    # output_feature = tf.stack(output_feature_list, axis=0)
    # return output_feature

    # result = Lambda(lambda x: tf.math.top_k(x, k=channel))(excitation)  # Get top K result and indices
    # index = Lambda(lambda x: x.indices)(result)  # indices of weight, from big weight's index to small
    # split_feature = Lambda(lambda x: tf.unstack(x, num=128, axis=0))(se_feature)
    # split_index = Lambda(lambda x: tf.unstack(x, num=128, axis=0))(index)
    # output_feature_list = list()
    # output_feature_list = Lambda(lambda x: x[0].append(tf.gather(x[1][i], x[2][i], axis=-1) for i in range(128)))([output_feature_list, split_feature, split_index])
    # output_feature = Lambda(lambda x: tf.stack(x, axis=0))(output_feature_list)

    result = Lambda(lambda x: tf.math.top_k(x, k=channel))(excitation)  # Get top K result and indices
    index = Lambda(lambda x: x.indices)(result)  # indices of weight, from big weight's index to small
    index_list = Lambda(lambda x: x[..., 0:int(channel * 0.5)])(index)
    se_feature = layers.Reshape((channel, 32*32))(se_feature)
    output_feature = Lambda(lambda x: tf.gather(x[0], x[1], axis=[0, 1]))([se_feature, index_list])

    # output_feature = Lambda(lambda x: tf.gather(x[0], x[1]))([se_feature, index_list])
    return output_feature

def ca_block(input_feature, reduction=ratio):
    channel = input_feature.shape.as_list()[-1]
    squeeze = layers.GlobalAveragePooling2D()(input_feature)
    excitation = layers.Dense(channel // reduction, activation='relu')(squeeze)
    excitation = layers.Dense(channel, activation='sigmoid')(excitation)
    excitation = layers.Reshape((1, 1, channel))(excitation)

    return layers.Multiply()([input_feature, excitation])

def sk_block(input_feature, M=2, r=ratio, L=32, G=1):
    inputs_shape = input_feature.get_shape().as_list()
    b, h, w = inputs_shape[0], inputs_shape[1], inputs_shape[2]
    filters = inputs_shape[-1]
    d = max(filters // r, L)
    x = input_feature

    xs = []

    for m in range(1, M + 1):
        if G == 1:
            _x = layers.Conv2D(filters, 3, dilation_rate=m, padding='same', use_bias=False)(x)
        else:
            c = filters // G
            _x = layers.DepthwiseConv2D(3, dilation_rate=m, depth_multiplier=c, padding='same', use_bias=False)(x)

            _x = layers.Reshape([h, w, G, c, c])(_x)
            _x = layers.Lambda(lambda x: tf.reduce_sum(_x, axis=-1))(_x)
            _x = layers.Reshape([h, w, filters])(_x)

        _x = layers.BatchNormalization()(_x)
        _x = layers.Activation('relu')(_x)

        xs.append(_x)

    U = layers.Add()(xs)
    s = layers.Lambda(lambda x: tf.reduce_mean(x, axis=[1, 2], keepdims=True))(U)
    z = layers.Conv2D(filters=d, kernel_size=1)(s)
    z = layers.BatchNormalization()(z)
    z = layers.Activation('relu')(z)

    x = layers.Conv2D(filters * M, 1)(z)
    x = layers.Reshape([1, 1, filters, M])(x)
    scale = layers.Softmax()(x)

    x = layers.Lambda(lambda x: tf.stack(x, axis=-1))(xs)  # b, h, w, c, M
    x = layers.Lambda(lambda x: tf.multiply(x[0], x[1]))([scale, x])
    x = layers.Lambda(lambda x: tf.reduce_sum(x, axis=-1))(x)

    return x

def cbam_block(cbam_feature, reduction=ratio):
    """Contains the implementation of Convolutional Block Attention Module(CBAM) block.
    As described in https://arxiv.org/abs/1807.06521.
    """

    cbam_feature = channel_attention(cbam_feature, reduction)
    cbam_feature = spatial_attention(cbam_feature)
    return cbam_feature

def channel_attention(input_feature, ratio=4):
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    channel = input_feature._keras_shape[channel_axis]

    shared_layer_one = layers.Dense(channel // ratio,activation='relu',
                                    kernel_initializer='he_normal',
                                    use_bias=True,
                                    bias_initializer='zeros')
    shared_layer_two = layers.Dense(channel,kernel_initializer='he_normal',
                                    use_bias=True,
                                    bias_initializer='zeros')

    avg_pool = layers.GlobalAveragePooling2D()(input_feature)
    avg_pool = layers.Reshape((1, 1, channel))(avg_pool)
    assert avg_pool._keras_shape[1:] == (1, 1, channel)
    avg_pool = shared_layer_one(avg_pool)
    assert avg_pool._keras_shape[1:] == (1, 1, channel // ratio)
    avg_pool = shared_layer_two(avg_pool)
    assert avg_pool._keras_shape[1:] == (1, 1, channel)

    max_pool = layers.GlobalMaxPooling2D()(input_feature)
    max_pool = layers.Reshape((1, 1, channel))(max_pool)
    assert max_pool._keras_shape[1:] == (1, 1, channel)
    max_pool = shared_layer_one(max_pool)
    assert max_pool._keras_shape[1:] == (1, 1, channel // ratio)
    max_pool = shared_layer_two(max_pool)
    assert max_pool._keras_shape[1:] == (1, 1, channel)

    cbam_feature = layers.Add()([avg_pool, max_pool])
    cbam_feature = layers.Activation('sigmoid')(cbam_feature)

    if K.image_data_format() == "channels_first":
        cbam_feature = layers.Permute((3, 1, 2))(cbam_feature)

    return layers.multiply([input_feature, cbam_feature])

def spatial_attention(input_feature):
    kernel_size = 7

    if K.image_data_format() == "channels_first":
        channel = input_feature._keras_shape[1]
        cbam_feature = layers.Permute((2, 3, 1))(input_feature)
    else:
        channel = input_feature._keras_shape[-1]
        cbam_feature = input_feature

    avg_pool = layers.Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(cbam_feature)
    assert avg_pool._keras_shape[-1] == 1
    max_pool = layers.Lambda(lambda x: K.max(x, axis=3, keepdims=True))(cbam_feature)
    assert max_pool._keras_shape[-1] == 1
    concat = layers.Concatenate(axis=3)([avg_pool, max_pool])
    assert concat._keras_shape[-1] == 2
    cbam_feature = layers.Conv2D(filters=1,
                                 kernel_size=kernel_size,
                                 strides=1,
                                 padding='same',
                                 activation='sigmoid',
                                 kernel_initializer='he_normal',
                                 use_bias=False)(concat)
    assert cbam_feature._keras_shape[-1] == 1

    if K.image_data_format() == "channels_first":
        cbam_feature = layers.Permute((3, 1, 2))(cbam_feature)

    return layers.multiply([input_feature, cbam_feature])


def danet_block(input_feature):

    pam_feature = pam_block(input_feature,ratio)
    cam_feature = cam_block(input_feature)
    danet_feature = layers.Add()([pam_feature, cam_feature])
    return danet_feature


def pam_block(input_feature,reduction):

    inputs_shape = input_feature.get_shape().as_list()
    n, h, w, c = inputs_shape[0], inputs_shape[1], inputs_shape[2],inputs_shape[3]
    b_conv = layers.Conv2D(int(c)//reduction, kernel_size=(1, 1), strides=(1, 1))(input_feature)
    c_conv = layers.Conv2D(int(c)//reduction, kernel_size=(1, 1), strides=(1, 1))(input_feature)
    d_conv = layers.Conv2D(int(c), kernel_size=(1, 1), strides=(1, 1))(input_feature)

    b_conv = layers.Reshape((h*w, int(c)//reduction))(b_conv)
    b_conv = layers.Permute((2,1))(b_conv)
    c_conv = layers.Reshape((h*w, int(c)//reduction))(c_conv)
    d_conv = layers.Reshape((h*w, c))(d_conv)

    attention = layers.Lambda(lambda x: tf.matmul(x[0], x[1]))([c_conv, b_conv])
    attention = layers.Activation('softmax')(attention)
    attention = layers.Lambda(lambda x: tf.matmul(x[0], x[1]))([attention, d_conv])
    attention = layers.Reshape((h, w, c))(attention)
    out = layers.Add()([attention, input_feature])
    return out


def cam_block(input_feature):

    inputs_shape = input_feature.get_shape().as_list()
    n, h, w, c = inputs_shape[0], inputs_shape[1], inputs_shape[2], inputs_shape[3]

    b_conv = layers.Reshape((h * w, c))(input_feature)
    b_conv = layers.Permute((2, 1))(b_conv)
    c_conv = layers.Reshape((h * w, c))(input_feature)
    d_conv = layers.Reshape((h * w, c))(input_feature)
    d_conv = layers.Permute((2, 1))(d_conv)

    attention = layers.Lambda(lambda x: tf.matmul(x[0], x[1]))([b_conv, c_conv])
    attention = layers.Activation('softmax')(attention)
    attention = layers.Lambda(lambda x: tf.matmul(x[0], x[1]))([attention, d_conv])
    attention = layers.Permute((2, 1))(attention)
    attention = layers.Reshape((h, w, c))(attention)
    out = layers.Add()([attention, input_feature])
    return out


def eca_block(input_feature, k_size = 3, gamma = 2, b = 1):
    """
    ECA-NET
    :param input_feature: input_feature.shape=[batchsize,h,w,channels]
    :param num:
    :param gamma:
    :param b:
    :return:
    """
    channels = K.int_shape(input_feature)[-1]
    t = int(abs((math.log(channels,2)+b)/gamma))
    k = t if t%2 else t+1
    x_global_avg_pool = GlobalAveragePooling2D()(input_feature)
    x = Reshape((channels,1))(x_global_avg_pool)
    x = Conv1D(1,kernel_size=k_size,padding="same")(x)
    x = Activation('sigmoid', name='eca_conv1_relu')(x)  #shape=[batch,chnnels,1]
    x = Reshape((1, 1, channels))(x)
    output = multiply([input_feature,x])
    return output