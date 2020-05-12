import numpy as np
import tensorflow as tf
import cv2

from script.data_operate.handle_data import HandleData
from script.train_face import Trainer

classes = ['trump', 'jay']
unprocessed_img_paths = '/home/maxingpei/AI-FakeFace/train_face'
processed_img_paths = '/home/maxingpei/AI-FakeFace/train_face'

face_xml = cv2.CascadeClassifier(
    '/home/maxingpei/AI-FakeFace/haarface_data/haarcascades_cuda/haarcascade_frontalface_default.xml')

epochs = 8000
batch_size = 16
images  = []

if __name__ == '__main__':
    handle_data = HandleData(classes)
    trainer = Trainer(epochs=epochs, batch_size=batch_size, classes=classes)

    # 将未处理的图片进行人脸提取并保存
    handle_data.extract_face(unprocessed_img_paths, processed_img_paths, face_xml)

    # 获取处理的图片路径
    trump_images_paths = handle_data.get_image_paths(processed_img_paths + '/' + 'trump' + '/' + 'img_data')
    jay_images_paths = handle_data.get_image_paths(processed_img_paths + '/' + 'jay' + '/' + 'img_data')

    # 加载处理的图片
    trump_images = handle_data.load_images(trump_images_paths) / 255.0
    jay_images = handle_data.load_images(jay_images_paths) / 255.0

    trainer.train(images_A=trump_images,
                  images_B=jay_images)





