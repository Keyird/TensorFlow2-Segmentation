import tensorflow as tf
from tensorflow.keras import layers, models
from networks.vggnet import vggnet_encoder

# 对应的输入输出形状：batch_size, height, width, channels
IMAGE_ORDERING = 'channels_last'

# 解码器
def decoder(feature_input, n_classes, n_upSample):
    # feature_input是vggnet第四个卷积块的输出特征矩阵
    # 26,26,512
    output = (layers.ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(feature_input)
    output = (layers.Conv2D(512, (3, 3), padding='valid', data_format=IMAGE_ORDERING))(output)
    output = (layers.BatchNormalization())(output)

    # 进行一次UpSampling2D，此时hw变为原来的1/8
    # 52,52,256
    output = (layers.UpSampling2D((2, 2), data_format=IMAGE_ORDERING))(output)
    output = (layers.ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(output)
    output = (layers.Conv2D(256, (3, 3), padding='valid', data_format=IMAGE_ORDERING))(output)
    output = (layers.BatchNormalization())(output)

    # 进行一次UpSampling2D，此时hw变为原来的1/4
    # 104,104,128
    for _ in range(n_upSample - 2):
        output = (layers.UpSampling2D((2, 2), data_format=IMAGE_ORDERING))(output)
        output = (layers.ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(output)
        output = (layers.Conv2D(128, (3, 3), padding='valid', data_format=IMAGE_ORDERING))(output)
        output = (layers.BatchNormalization())(output)

    # 进行一次UpSampling2D，此时hw变为原来的1/2
    # 208,208,64
    output = (layers.UpSampling2D((2, 2), data_format=IMAGE_ORDERING))(output)
    output = (layers.ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(output)
    output = (layers.Conv2D(64, (3, 3), padding='valid', data_format=IMAGE_ORDERING))(output)
    output = (layers.BatchNormalization())(output)

    # 此时输出为h_input/2,w_input/2,nclasses
    # 208,208,2
    output = layers.Conv2D(n_classes, (3, 3), padding='same', data_format=IMAGE_ORDERING)(output)

    return output


# 语义分割网络SegNet
def SegNet(input_height=416, input_width=416, n_classes=2, n_upSample=3, encoder_level=3):

    img_input, features = vggnet_encoder(input_height=input_height, input_width=input_width)
    feature = features[encoder_level]  # (26,26,512)
    output = decoder(feature, n_classes, n_upSample)

    # 将结果进行reshape
    output = tf.reshape(output, (-1, int(input_height / 2) * int(input_width / 2), 2))
    output = layers.Softmax()(output)

    model = tf.keras.Model(img_input, output)

    return model


