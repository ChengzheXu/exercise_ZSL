import tensorflow as tf
from IndirectAttribute_VGG19 import IndirectAttribute_VGG19 as IAP
import time
import numpy as np

dropout_rate = 0.1
batch_size = 2
epochs = 100
learning_rate = (1e-3)
regul = 0.0

img_width, img_height, img_path = 224, 224, 3
attribute_length = 10
num_classes_train = 5
num_classes_all = 10

attri_list_all = np.array([])
attri_list_train = np.array([])

train_loss_list = []
train_acc_list = []
vali_acc_list = []

model = IAP(learning_rate=learning_rate,
            attribute_length=attribute_length,
            num_classes_all=num_classes_all,
            num_classes_train=num_classes_train,
            attri_list_all=attri_list_all,
            attri_list_train=attri_list_train,
            vgg19_npy_path="./vgg19.npy",
            trainable=False,
            img_shape=224,
            regul=regul
            )

init_op = tf.global_variables_initializer()

with tf.Session() as sess:

    sess.run(init_op)

    for i in range(epochs):

        print("epoch", i+1, ":")
        img_tensor_train, img_attribute_train, img_label_train_train, img_label_train_all = \
            Get_next_batch("train", batch_size)

        img_tensor_vali, img_attribute_vali, img_label_vali_train,  img_label_vali_all = \
            Get_next_batch("validation", batch_size)

        print("Training ...", end=' ')

        start_time = time.time()

        # train
        [_, train_loss, train_acc] = sess.run([model.train_op, model.loss, model.acc],
                                              feed_dict={model.img_tensor: img_tensor_train,
                                                         model.img_attribute: img_attribute_train,
                                                         model.img_label_all: img_label_train_all,
                                                         model.img_label_train: img_label_train_train,
                                                         model.dropout: dropout_rate})

        end_time = time.time()

        vali_acc = sess.run(model.acc, feed_dict={model.img_tensor: img_tensor_vali,
                                                  model.img_attribute: img_attribute_vali,
                                                  model.img_label_all: img_label_vali_all,
                                                  model.img_label_train: img_label_vali_train,
                                                  model.dropout: 0.0})

        print("Done.")
        print("Cost ", end_time - start_time, "Train loss=", train_loss,
              "Train accuracy=", train_acc, "Validation accuracy=", vali_acc)
        print("#" * 30)

        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        vali_acc_list.append(vali_acc)

    print("Finished.")

def Get_next_batch(Train_or_Vali, batch_size):
    """
    Get next batch according to size.
    :param Train_or_Vali: Whether to train or validate the model.
    :param batch_size: The size of the training batch.
    :return: training data, shape = [batch_size, img_width, img_height, img_path],
    attribute, shape = [batch_size, attribute_length]
    and label(one-hot), shape = [batch_size, num_classes]
    for next batch, type = np.array.
    """

    return img_batch, attribute_batch, label_train_batch, label_all_batch