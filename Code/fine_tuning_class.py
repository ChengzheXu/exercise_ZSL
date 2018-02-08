#!/usr/bin/env python
# _*_ coding: utf-8 _*_

# 参考资料 http://blog.csdn.net/caanyee/article/details/52502759

import tensorflow as tf
# # config  = tf.ConfigProto()
# # config.gpu_options.allow_growth = True
# # session = tf.Session(config=config)

config = tf.ConfigProto() 
config.gpu_options.per_process_gpu_memory_fraction = 0.9 # 占用GPU90%的显存 
session = tf.Session(config=config)

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"

from keras.models import Model
from keras.layers import Dense, Dropout
import keras.optimizers as optimizer
from keras.preprocessing.image import ImageDataGenerator

# 图片大小 注意网络中有pooling，会造成图片缩小
img_width, img_height = 224, 224
# 类别总数
num_class = 50

# 训练的batch_size
batch_size = 16
# 训练的epoch
epochs = 100

# 全链接网络网络结构
fc1 = 256
fc2 = 128
dropout_rate = 0.5
lr = (1e-2) # fine-tuning 时应该保证learninig rate很小
momentum = 0.9

class FineTuning(object):
    """
    The Fine Tune class.
    """
    def __init__(self, training_data_filename, validation_data_file_name,
                 base_model="VGG19"):
        """
        :param training_data_filename: The directory name of training data.
        :param validation_data_file_name: The directory name of validation data.
        :param base_model: The pre-trained model.
        """

        self.train_datagen = ImageDataGenerator(
            rescale=1.0 / 255,  # 有可能需要用到rescale
            rotation_range=5,  # 图片旋转
            width_shift_range=0.1,  # 水平移动
            height_shift_range=0.1,  # 竖直移动
            zoom_range=0.2,  # 随机放大
            horizontal_flip=True,  # 水平翻转
            fill_mode='nearest'  # 最近邻方法补全像素
        )

        self.train_generator = self.train_datagen.flow_from_directory(training_data_filename,
                                                                      target_size=(img_width, img_height),
                                                                      batch_size=batch_size)

        self.validation_datagen = ImageDataGenerator(rescale=1.0 / 255  # 有可能需要用到rescale
                                                   )
        self.validation_generator = self.validation_datagen.flow_from_directory(validation_data_file_name,
                                                                           target_size=(img_width, img_height),
                                                                           batch_size=batch_size)

        if base_model == "InceptionV3":
            from keras.applications.inception_v3 import InceptionV3
            self.base_model = InceptionV3(weights='imagenet', include_top=False, pooling='avg')
        elif base_model == "ResNet50":
            from keras.applications.resnet50 import ResNet50
            self.base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
        elif base_model == "VGG16":
            from keras.applications.vgg16 import VGG16
            self.base_model = VGG16(weights='imagenet', include_top=False, pooling='avg')
        else:
            from keras.applications.vgg19 import VGG19
            self.base_model = VGG19(weights='imagenet', include_top=False, pooling='avg')

        h_fc1 = Dense(fc1, activation='relu', use_bias=True)(self.base_model.output)
        h_fc1_drop = Dropout(rate=dropout_rate)(h_fc1)

        h_fc2 = Dense(fc2, activation='relu', use_bias=True)(h_fc1_drop)
        h_fc2_drop = Dropout(rate=dropout_rate)(h_fc2)

        predictions = Dense(num_class, activation='softmax', use_bias=True)(h_fc2_drop)

        self.model = Model(inputs=self.base_model.input, outputs=predictions)

        print("Train the fully connected layers.")
        self.fine_tuning(23)
        print("Done.")
        print("Fine Turning the architecture.")
        self.fine_tuning(17)
        self.fine_tuning(12)
        print("Done.")
        # for VGG19: 23, 17, 12
        # for VGG16: 20, 15, 11
        # for InceptionV3: 312, 280, 249
        # for ResNet50: 175, 163, 153

        # Use the network
        # Use the network

    def fine_tuning(self, frozen_range):
        """
        Fine Tune the model, with the model's layers in range(frozen_range) to be set frozen.
        :param frozen_range: The range of frozen layers.
        :return: None.
        """
        for layer in self.model.layers[:frozen_range:]:
            layer.trainable = False

        self.model.compile(optimizer=optimizer.SGD(lr=lr, momentum=momentum),
                           loss='categorical_crossentropy',
                           metrics=['accuracy'],)

        self.model.fit_generator(self.train_generator,
                                 steps_per_epoch=self.train_generator.samples // batch_size,
                                 epochs=epochs,
                                 verbose=1,
                                 validation_data=self.validation_generator,
                                 validation_steps=self.validation_generator.samples // batch_size,
                                 )

        # self.model.fit_generator(self.train_generator,
        #                          samples_per_epoch=2000,
        #                          nb_epoch=50,
        #                          validation_data=self.validation_generator,
        #                          nb_val_samples=800
        #                          )

        return None


fine_tuning_model = FineTuning("./Data/train", "./Data/validation")
print("Validation Finished.")
