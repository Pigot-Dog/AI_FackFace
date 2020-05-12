from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf

from script.network_setting import Net_Setting
from script.data_operate.handle_data import HandleData

class Trainer(object):
    def __init__(self, lr=5e-5, beta_1=0.5, beta_2=0.999, epochs=1000,
                 batch_size=32, image_shape=[64, 64, 3], encoder_dim=1024, classes=[], **kwargs):
        self.lr = lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epochs = epochs
        self.batch_size = batch_size
        self.image_shape = image_shape
        self.encoder_dim = encoder_dim
        self.classes = classes

        self.net_setting = Net_Setting(image_shape=self.image_shape,
                                       encoder_dim=self.encoder_dim,
                                       **kwargs)

        self.handle_data = HandleData(self.classes)

        self.encoder = None
        self.decoder_A = None
        self.decoder_B = None

    def train(self, images_A, images_B):
        autoencoder_A, autoencoder_B = self.build()

        print("开始训练...")

        for epoch in range(self.epochs):
            print("第{}代,开始训练...".format(epoch))
            warped_A, target_A = self.get_training_data(images_A, self.batch_size)
            warped_B, target_B = self.get_training_data(images_B, self.batch_size)

            loss_A = autoencoder_A.train_on_batch(warped_A, target_A)
            loss_B = autoencoder_B.train_on_batch(warped_B, target_B)
            print("loss_A{}, loss_B:{}".format(loss_A, loss_B))

            if epoch % 7990 == 0:
                test_A = warped_A[0:3]
                test_B = warped_B[0:3]

                print("开始预测 ...")
                figure_A = np.stack([
                    test_A,
                    autoencoder_A.predict(test_A),
                    autoencoder_B.predict(test_A),
                ], axis=1)
                figure_B = np.stack([
                    test_B,
                    autoencoder_B.predict(test_B),
                    autoencoder_A.predict(test_B),
                ], axis=1)

                figure = np.concatenate([figure_A, figure_B], axis=0)
                figure = figure.reshape((2, 3) + figure.shape[1:])
                figure = self.stack_images(figure)
                # 反归一化
                figure = np.clip(figure * 255, 0, 255).astype('uint8')
                # BGR -> RGB
                plt.imshow(cv2.cvtColor(figure, cv2.COLOR_BGR2RGB))
                plt.show()

            if epoch % 1000 == 0:
                self.save_model_weights()

        self.save_model_weights()

    # TODO: encoder and decoder
    def build(self):
        optimizer = Adam(self.lr, self.beta_1, self.beta_2)
        inputs = Input(shape=self.image_shape)

        encoder = self.net_setting.Encoder()
        decoder_A = self.net_setting.Decoder()
        decoder_B = self.net_setting.Decoder()

        self.encoder = encoder
        self.decoder_A = decoder_A
        self.decoder_B = decoder_B

        autoencoder_A = Model(inputs, decoder_A(encoder(inputs)))
        autoencoder_B = Model(inputs, decoder_B(encoder(inputs)))

        autoencoder_A.compile(optimizer=optimizer, loss='mean_absolute_error')
        autoencoder_B.compile(optimizer=optimizer, loss='mean_absolute_error')

        return autoencoder_A, autoencoder_B

    def get_training_data(self, images, batch_size):
        indices = np.random.randint(len(images), size=batch_size)
        for i, index in enumerate(indices):
            image = images[index]

            # 将图片进行预处理
            image = self.handle_data.random_transform(image)
            warped_img, target_img = image, image
            if i == 0:
                warped_images = np.empty(
                    (batch_size,) + warped_img.shape, warped_img.dtype)
                target_images = np.empty(
                    (batch_size,) + target_img.shape, target_img.dtype)

            warped_images[i] = warped_img
            target_images[i] = target_img

        return warped_images, target_images

    # 保存模型
    def save_model_weights(self):
        directory_paths = "/home/maxingpei/AI-FakeFace/save_model"
        self.encoder.save_weights(directory_paths + "/encoder.h5")
        self.decoder_A.save_weights(directory_paths + "/decoder_A.h5")
        self.decoder_B.save_weights(directory_paths + "/decoder_B.h5")
        print("模型保存完毕")

    # 加载模型
    def load_model_weights(self, directory):
        self.encoder.load_weights(directory + "/encoder.h5")
        self.decoder_A.load_weights(directory + "/decoder_A.h5")
        self.decoder_B.load_weights(directory + "/decoder_B.h5")

        return self.encoder, self.decoder_A, self.decoder_B

    def get_transpose_axes(self, n):
        if n % 2 == 0:
            y_axes = list(range(1, n-1, 2))
            x_axes = list(range(0, n-1, 2))
        else:
            y_axes = list(range(0, n-1, 2))
            x_axes = list(range(1, n-1, 2))

        return y_axes, x_axes, [n-1]

    # 将多幅图片拼接成一幅
    def stack_images(self, images):
        images_shape = np.array(images.shape)
        # 得到三个列表:[0, 2], [1, 3], [4]
        new_axes = self.get_transpose_axes(len(images_shape))
        new_shape = [np.prod(images_shape[x]) for x in new_axes]

        return np.transpose(images, axes=np.concatenate(new_axes)).reshape(new_shape)


