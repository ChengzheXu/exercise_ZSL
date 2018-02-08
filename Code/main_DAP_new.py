import tensorflow as tf
from DirectAttribute_VGG19 import DirectAttribute_VGG19 as DAP
import time
import numpy as np
import image_data_handle as ImgH
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

dropout_rate = 0.1
batch_size = 16
epochs = 100
learning_rate = (1e-3)
regul = 0.0

attribute_length = 85
num_classes_all = 25
attri_list_all = ImgH.attribute_list_all

print(attri_list_all)

train_loss_list = []
vali_loss_list = []
train_acc_list = []
vali_acc_list = []

img_tensor_train, img_attribute_train, img_label_train = ImgH.Get_next_batch(
    "train", batch_size, 2)

print(type(img_tensor_train[0][0][0][0]))
print(np.shape(img_tensor_train))


model = DAP(learning_rate=learning_rate,
            attribute_length=attribute_length,
            num_classes_all=num_classes_all,
            attri_list_all=attri_list_all.astype('float32'),
            batch_size=batch_size,
            vgg19_npy_path="./vgg19.npy",
            trainable=False,
            img_shape=224, regul=regul)

print("*" * 30)
init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)

    for i in range(epochs):

        print("epoch", i + 1, ":")
        # img_tensor_train, img_attribute_train, _, img_label_train = ImgH.Get_next_batch("train", batch_size)
        # img_tensor_train, img_attribute_train, img_label_train = ImgH.Get_next_batch("train", batch_size, i, use_word2vec = 0, attribute = 'bi')
        img_tensor_train, img_attribute_train, img_label_train = ImgH.Get_next_batch(
            "train", batch_size, i)

        img_tensor_train = img_tensor_train.astype('float32')
        img_attribute_train = img_attribute_train.astype('float32')
        img_label_train = img_label_train.astype('float32')

        # img_tensor_vali, img_attribute_vali, _, img_label_vali = Img.Get_next_batch("validation", batch_size)
        # img_tensor_vali, img_attribute_vali, img_label_vali = ImgH.Get_next_batch("validation", batch_size, i, use_word2vec = 0, attribute = 'bi')
        img_tensor_vali, img_attribute_vali, img_label_vali = ImgH.Get_next_batch(
            "validation", batch_size, i)

        img_tensor_vali = img_tensor_vali.astype('float32')
        img_attribute_vali = img_attribute_vali.astype('float32')
        img_label_vali = img_label_vali.astype('float32')

        print("Training ...", end=' ')

        start_time = time.time()

        # train
        [_, train_loss, train_acc] = sess.run([model.train_op, model.loss, model.acc],
                                              feed_dict={model.img_tensor: img_tensor_train,
                                                         model.img_attribute: img_attribute_train,
                                                         model.img_label_all: img_label_train,
                                                         model.dropout: dropout_rate})

        end_time = time.time()

        [vali_loss, vali_acc] = sess.run([model.loss, model.acc],
                                         feed_dict={model.img_tensor: img_tensor_vali,
                                                    model.img_attribute: img_attribute_vali,
                                                    model.img_label_all: img_label_vali,
                                                    model.dropout: 0.0})

        print("Done.")
        print("Cost ", end_time - start_time, "Train loss=", train_loss, "Validation loss=", vali_loss,
              "Train accuracy=", train_acc, "Validation accuracy=", vali_acc)
        print("#" * 30)

        train_loss_list.append(train_loss)
        vali_loss_list.append(vali_loss)
        train_acc_list.append(train_acc)
        vali_acc_list.append(vali_acc)

    print("Finished.")
