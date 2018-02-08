import tensorflow as tf
from IndirectAttribute_VGG19_weak import IndirectAttribute_VGG19 as IAP
import time
import numpy as np
import image_data_handle as ImgH
import os

def Get_label(attribute, label_attri):
    """
    Get the labels according to attributes.
    :param attribute: The attributes (tf.tensor), shape = [batch_size, attribute_length].
    :param label_attri: The list of attributes (np.array), shape = [num_classes, attribute_length].
    Caution: the attributes must be in the order according to the labels.
    :return: The relative labels (one-hot, tf.tensor), shape = [batch_size, num_classes].
    """
    # with tf.Session() as sess:
    #     attribute = attribute.eval(session=sess)

    # cosine距离

    label_idx = list(np.argmax(np.dot(attribute, label_attri.T), axis=1))

    label = np.zeros([np.shape(attribute)[0], np.shape(label_attri)[0]])

    for instance in range(np.shape(label)[0]):
        label[instance][label_idx[instance]] = 1

    return label

def acc_label(img_label, predic_label):
    right = 0

    img_label_ = np.argmax(img_label, axis=1)
    predic_label_ = np.argmax(predic_label, axis=1)

    for instance in range(np.shape(img_label_)[0]):
        if img_label_[instance] == predic_label_[instance]:
            right += 1

    return right/np.shape(img_label_)[0]


os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

dropout_rate = 0.1
batch_size = 32
epochs = 100
learning_rate = (1e-4)

attribute_length = 85
num_classes_all = 25
attri_list_all = ImgH.attribute_list_all

train_loss_list = []
vali_loss_list = []
train_acc_list = []
vali_acc_list = []

img_tensor_train, img_attribute_train, img_label_train = ImgH.Get_next_batch(
    "train", batch_size, 2)

model = IAP(learning_rate=learning_rate,
            attribute_length=attribute_length,
            num_classes_all=num_classes_all,
            attri_list_all=attri_list_all.astype('float32'),
            batch_size=batch_size,
            vgg19_npy_path="./vgg19.npy",
            trainable=False,
            img_shape=224)

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
        [_, train_loss, train_predic_attr] = sess.run([model.train_op, model.loss, model.predic_attr],
                                              feed_dict={model.img_tensor: img_tensor_train,
                                                         model.img_attribute: img_attribute_train,
                                                         model.img_label_all: img_label_train,
                                                         model.dropout: dropout_rate})

        end_time = time.time()

        predic_label_train = Get_label(train_predic_attr, attri_list_all)
        train_acc = acc_label(img_label_train, predic_label_train)

        [vali_loss, test_predic_attr] = sess.run([model.loss, model.predic_attr],
                                         feed_dict={model.img_tensor: img_tensor_vali,
                                                    model.img_attribute: img_attribute_vali,
                                                    model.img_label_all: img_label_vali,
                                                    model.dropout: 0.0})

        predic_label_test = Get_label(test_predic_attr, attri_list_all)
        vali_acc = acc_label(img_label_vali, predic_label_test)

        print("Done.")
        print("Cost ", end_time - start_time, "Train loss=", train_loss, "Validation loss=", vali_loss,
              "Train accuracy=", train_acc, "Validation accuracy=", vali_acc)
        print("#" * 30)

        train_loss_list.append(train_loss)
        vali_loss_list.append(vali_loss)
        train_acc_list.append(train_acc)
        vali_acc_list.append(vali_acc)

    print("Finished.")
