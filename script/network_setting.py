import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Conv2D, LeakyReLU, Input, Dense, Flatten, Reshape
from tensorflow.keras.models import Model


# 子像素卷积
class PixelShuffler():
    def __init__(self, size=(2, 2), data_format=None, **kwargs):
        self.size = size
        self.data_format = data_format

    def __call__(self, inputs):
        batch_size, h, w, c = inputs.shape

        if batch_size is None:
            batch_size = -1
        rh, rw = self.size

        # 计算转换后图层大小和通道数
        oh, ow = h * rh, w * rw
        oc = c // (rh * rw)

        outputs = tf.reshape(inputs, (batch_size, h, w, rh, rw, oc))
        outputs = tf.transpose(outputs, [0, 1, 3, 2, 4, 5])
        outputs = tf.reshape(outputs, (batch_size, oh, ow, oc))

        return outputs

    def get_config(self):
        config_dict = {
            'size': self.size,
            'data_format': self.data_format
        }
        return config_dict


class Net_Setting(object):
    def __init__(self, image_shape=[64, 64, 3], encoder_dim=1024, **kwargs):
        self.image_shape = image_shape
        self.encoder_dim = encoder_dim
        self.pixel_shuffler = PixelShuffler()

    # 编码器
    def Encoder(self):
        input_ = Input(shape=self.image_shape)
        x = input_                                # x.shape = (None, 64, 64, 3)
        x = self._conv(inputs=input_, filters=128)
        x = self._conv(inputs=x, filters=256)
        x = self._conv(inputs=x, filters=512)
        x = self._conv(inputs=x, filters=1024)     # x.shape = (None, 4, 4, 1024)
        x = Dense(self.encoder_dim)(Flatten()(x))  # x.shape = (None, 1024)
        x = Dense(4*4*1024)(x)
        x = Reshape((4, 4, 1024))(x)               # x.shape = (None, 4, 4, 1024)
        x = self._upscale(inputs=x, filters=512)   # x.shape = (None, 8, 8, 512)

        return Model(input_, x)

    # 解码器
    def Decoder(self):
        input_ = Input(shape=(8, 8, 512))
        x = input_
        x = self._upscale(inputs=x, filters=256)   # x.shape = (None, 16, 16, 256)
        x = self._upscale(inputs=x, filters=128)   # x.shape = (None, 32, 32, 128)
        x = self._upscale(inputs=x, filters=64)    # x.shape = (None, 64, 64, 64)
        x = Conv2D(3, kernel_size=5, padding='same', activation='sigmoid')(x)  # x.shape = (None, 64, 64, 3)

        return Model(input_, x)

    # 下采样层
    # (n, n, c) -> (0.5n, 0.5n, filters)
    def _conv(self, inputs=None, filters=64, kernel_size=5, strides=2, padding='same'):
        outputs = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides,
                         padding=padding)(inputs)
        outputs = LeakyReLU(0.1)(outputs)

        return outputs

    # 上采样
    # (n, n, c) -> (n, n, 4*filters) -> (2n, 2n, filters)
    def _upscale(self, inputs=None, filters=64, kernel_size=3, strides=1, padding='same'):
        outputs = Conv2D(filters=filters*4, kernel_size=kernel_size, padding=padding)(inputs)
        outputs = LeakyReLU(0.1)(outputs)
        outputs = self.pixel_shuffler(outputs)

        return outputs



