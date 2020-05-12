import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2

from script.train_face import Trainer
from script.data_operate.handle_data import HandleData
# from script.example.run_face import unprocessed_img_paths, processed_img_paths


trump_test_img_path = "/home/maxingpei/AI-FakeFace/test_face/target/trump.png"
jay_test_img_path = "/home/maxingpei/AI-FakeFace/test_face/src/jay.jpeg"

save_result = "/home/maxingpei/AI-FakeFace/save_result"
classes = ['trump', 'jay']
load_model_directory = "/home/maxingpei/AI-FakeFace/save_model"
batch_size = 32

face_xml = cv2.CascadeClassifier(
    '/home/maxingpei/AI-FakeFace/haarface_data/haarcascades_cuda/haarcascade_frontalface_default.xml')

if __name__ == '__main__':
    trainer = Trainer(classes)
    handle_data = HandleData(classes)
    print("开始加载模型...")
    autoencoder_A, autoencoder_B = trainer.build()
    encoder, decoder_A, decoder_B = trainer.load_model_weights(load_model_directory)

    # 加载处理的图片
    jay_image = cv2.imread(jay_test_img_path)
    jay_image = cv2.resize(jay_image, (256, 256))
    trump_image = cv2.imread(trump_test_img_path)
    trump_image = cv2.resize(trump_image, (256, 256))

    # # TODO： test
    # plt.imshow(cv2.cvtColor(trump_image, cv2.COLOR_BGR2RGB))
    # plt.show()

    assert jay_image.shape == (256, 256, 3) and trump_image.shape == (256, 256, 3)

    # 提取人脸
    jay_gray = cv2.cvtColor(jay_image, cv2.COLOR_BGR2GRAY)
    jay_face = face_xml.detectMultiScale(jay_gray, 1.3, 4)
    if len(jay_face) == 1:
        for (x, y, w, h) in jay_face:
            jay_roi_face = jay_image[y:y + h, x:x + w]
            jay_roi_face = cv2.resize(jay_roi_face, (64, 64), 0, 0, cv2.INTER_LINEAR)

            # # TODO： test
            # plt.imshow(cv2.cvtColor(jay_roi_face, cv2.COLOR_BGR2RGB))
            # plt.show()

    trump_gray = cv2.cvtColor(trump_image, cv2.COLOR_BGR2GRAY)
    trump_face = face_xml.detectMultiScale(trump_gray, 1.3, 4)
    assert len(trump_face) == 1

    # 预测前先阔维
    if len(jay_roi_face.shape) < 4:
        jay_roi_face = np.expand_dims(jay_roi_face, 0)

    # 预测
    new_jay_face = autoencoder_B.predict(jay_roi_face / 255.0)[0]
    # 图片反归一化并调整大小
    new_jay_face = np.clip(new_jay_face * 255, 0, 255).astype(trump_image.dtype)
    new_jay_face = cv2.resize(new_jay_face, (trump_face[0][3], trump_face[0][2]))

    # TODO： test
    plt.imshow(cv2.cvtColor(new_jay_face, cv2.COLOR_BGR2RGB))
    plt.show()

    # create an all white mask
    mask = 255 * np.ones(new_jay_face.shape, new_jay_face.dtype)
    height, width, channels = trump_image.shape
    center = (height // 2, width // 2)
    # 泊松融合的NORMAL_CLONE方法
    normal_clone = cv2.seamlessClone(new_jay_face, trump_image, mask, center, cv2.NORMAL_CLONE)
    # # 泊松融合的MIXED_CLONE方法
    # mixed_clone = cv2.seamlessClone(new_jay_face, trump_image, mask, center, cv2.MIXED_CLONE)

    # 显示
    plt.imshow(cv2.cvtColor(normal_clone, cv2.COLOR_BGR2RGB))
    plt.show()

    # 保存
    cv2.imwrite(save_result + "/" + "jay&trump_normal.jpg", normal_clone)
    # cv2.imwrite(mixed_clone, save_result + "/" + "jay&trump_mixed.jpg")

