import os
import cv2
import numpy as np
import glob


class HandleData(object):
    def __init__(self, classes=[], **kwargs):
        self.classes = classes

    # 获取图片路径
    # TODO: directory is absolute path
    def get_image_paths(self, directory):
        return [x.path for x in os.scandir(directory) if x.name.endswith(".jpg")
                or x.name.endswith(".png") or x.name.endswith(".jpeg")]

    # 加载图片
    def load_images(self, image_paths):
        iter_all_images = (cv2.imread(fn) for fn in image_paths)

        for i, image in enumerate(iter_all_images):
            if i == 0:
                all_images = np.empty((len(image_paths),) + image.shape, dtype=image.dtype)

            all_images[i] = image

        return all_images

    # 保存图片
    def save_image(self, images, directory):
        pass

    # 加载图片,提取人脸,并保存
    # TODO: load_directory and save_directory are absolute paths
    def extract_face(self, load_directory, save_directory, face_xml):
        print("读取训练图片...")
        for train_fields in self.classes:
            image_index = 0
            image_index = self.classes.index(train_fields)
            path = os.path.join(load_directory, train_fields, '*g')

            images_paths = self.get_image_paths(load_directory)

            train_files = glob.glob(path)

            print("正在处理 {} 的人脸训练图片".format(train_fields))
            for fl in train_files:
                srcimage = cv2.imread(fl)
                gray = cv2.cvtColor(srcimage, cv2.COLOR_BGR2GRAY)
                train_face = face_xml.detectMultiScale(gray, 1.3, 4)
                num_face = len(train_face)
                if num_face == 1:
                    for (x, y, w, h) in train_face:
                        roi_gray_face = gray[y:y + h, x:x + w]
                        roi_color_face = srcimage[y:y + h, x:x + w]
                        roi_color_face = cv2.resize(roi_color_face, (64, 64), 0, 0, cv2.INTER_LINEAR)
                        fileName = save_directory + '/' + train_fields + \
                                   '/' + 'img_data' + '/' + str(image_index) + '.jpg'
                        image_index = image_index + 1
                        cv2.imwrite(fileName, roi_color_face)
            print(" {} 的人脸训练图片处理完成".format(train_fields))

    # 数据增强
    def random_transform(self, image):
        h, w = image.shape[0:2]

        # 随机旋转角度,范围(-10, 10)
        rotation = np.random.uniform(-10, 10)
        # 随机缩放比例,范围(0.95, 1.05)
        scale = np.random.uniform(0.95, 1.05)
        # 随机平移距离,范围(-0.05, 0.05)
        tx = np.random.uniform(-0.05, 0.05) * w
        ty = np.random.uniform(-0.05, 0.05) * h
        # 定义放射变化矩阵
        mat = cv2.getRotationMatrix2D((w//2, h//2), rotation, scale)
        mat[:,2] += (tx, ty)
        # 进行放射变化
        result = cv2.warpAffine(
            image, mat, (w, h), borderMode=cv2.BORDER_REPLICATE)
        # 图片有40%的可能性被翻转
        if np.random.random() < 0.4:
            result = result[:, ::-1]

        return result


