import tensorflow as tf
import numpy as np
from functools import reduce

class IndirectWordVecDirectAttribute_VGG19(object):
    """
    A direct attribute learner, use VGG19.
    """
    def __init__(self, learning_rate, attribute_length, num_classes_all, num_classes_train, attri_list_all, WV_list_all,
                 attri_list_train, WV_list_train,
                 vgg19_npy_path=None,
                 trainable=False,
                 img_shape=224,
                 regul_DA=0.0,
                 regul_IW=0.0):
        """
        :param learning_rate: Learning rate, float.
        :param attribute_length: The length of attribute, int.
        :param num_classes_all: The number of classes in training and validation set, int.
        :param num_classes_train: The number of classes in training set, int.
        :param attri_list_all: The relationship of labels and attributes in training and validation set, the list of attributes (np.array),
        shape = [num_classes_all, attribute_length].
        Caution: the attributes must be in the order according to the labels.
        :param WV_list_all: The relationship of labels and word vectors in training and validation set, the list of attributes (np.array),
        shape = [num_classes_all, word_vector_length].
        Caution: the attributes must be in the order according to the labels.
        :param attri_list_train: The relationship of labels and attributes in training set, the list of attributes (np.array),
        shape = [num_classes_train, attribute_length].
        Caution: the attributes must be in the order according to the labels.
        :param WV_list_train: The relationship of labels and word vectors in training set, the list of attributes (np.array),
        shape = [num_classes_train, word_vector_length].
        Caution: the attributes must be in the order according to the labels.
        :param vgg19_npy_path: The path of "vgg19.npy".
        :param trainable: A bool tensor, whether the VGG is trainable.
        :param img_shape: The width or height of image, int.
        :param regul_DA: The rate of direct attribute learning regularization, positive float.
        :param regul_IW: The rate of indirect work vector learning regularization, positive float.
        """
        relu = tf.nn.relu
        tanh = tf.nn.tanh
        sigmoid = tf.nn.sigmoid
        BatchNormalization = tf.layers.batch_normalization
        dropout = tf.nn.dropout
        dense = tf.layers.dense
        softmax = tf.nn.softmax

        # 导入VGG19模型参数并存储。
        if vgg19_npy_path is not None:
            self.data_dict = np.load(vgg19_npy_path, encoding='latin1').item()
        else:
            self.data_dict = None

        self.var_dict = {}
        self.trainable = trainable
        self.learning_rate = learning_rate

        self.img_width = img_shape
        self.img_height = img_shape
        self.attribute_length = attribute_length
        self.num_classes_all = num_classes_all
        self.num_classes_train = num_classes_train
        self.attri_list_all = attri_list_all
        self.attri_list_train = attri_list_train
        self.WV_list_train = WV_list_train
        self.WV_list_all = WV_list_all

        self.img_tensor = tf.placeholder(tf.float32, shape=[None, self.img_width, self.img_height, 3])
        self.img_attribute = tf.placeholder(tf.float32, shape=[None, self.attribute_length])
        self.img_label_all = tf.placeholder(tf.float32, shape=[None, self.num_classes_all])
        self.img_label_train = tf.placeholder(tf.float32, shape=[None, self.num_classes_train])
        self.dropout = tf.placeholder(tf.float32)
        self.regul_DA = max(0., regul_DA)
        self.regul_IW = max(0., regul_IW)

        self.build(input_image=self.img_tensor, include_top=False)

        """
        后端网络结构。
        """

        # h_fc1 = dropout(sigmoid(BatchNormalization(dense(tf.reshape(self.pool5, [-1, 25088]),
        #                                                  self.attribute_length), axis=1,
        #                                            training=True)), keep_prob=1-self.dropout)

        self.feature = tf.reshape(self.pool5, [-1, 25088])

        # Direct Attribute Learning.
        self.predic_attr_ = dense(self.feature, self.attribute_length)

        self.predic_attr = sigmoid(self.predic_attr_)

        # Indirect WordVec learning.
        self.predic_label_train_ = dense(self.feature,
                                         self.num_classes_train)

        self.predic_label_train = softmax(self.predic_label_train_)

        self.predic_WV = self.Get_WV(self.predic_label_train, self.WV_list_train)

        # predict label.
        self.predic_label = self.Get_label(self.predic_attr, self.predic_WV, self.attri_list_all, self.WV_list_all)

        self.acc = self.acc_label(self.img_label_all, self.predic_label)

        # 仅在train时是有意义的。
        self.loss = (1.0/(1.0+self.regul_DA+self.regul_IW)) * tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=self.img_attribute, logits=self.predic_attr_)) +\
                    (self.regul_DA / (1.0+self.regul_DA+self.regul_IW)) * tf.reduce_mean((self.img_label_all-self.predic_label)**2)\
                    + (self.regul_IW / (1.0+self.regul_DA+self.regul_IW)) * tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            labels=self.img_label_train, logits=self.predic_label_train_))

        self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

    def build(self, input_image, include_top=False,  train_mode=None):
        """
        Load variable from .npy file to build the VGG19.
        :param input_image: RGB image tensor: [batch, height, width, 3]. Values scaled [0, 1].
        :param include_top: A bool tensor, whether to include the fully connected layers.
        :param train_mode: A bool tensor, usually a placeholder: if True, dropout will be turned on
        """

        VGG_MEAN = [103.939, 116.779, 123.68]

        input_image_scaled = input_image * 255.0

        # Convert RGB to BGR.
        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=input_image_scaled)

        assert red.get_shape().as_list()[1:] == [self.img_width, self.img_height, 1]
        assert green.get_shape().as_list()[1:] == [self.img_width, self.img_height, 1]
        assert blue.get_shape().as_list()[1:] == [self.img_width, self.img_height, 1]

        bgr_image = tf.concat(axis=3, values=[
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2],
        ])
        assert bgr_image.get_shape().as_list()[1:] == [224, 224, 3]

        self.conv1_1 = self.conv_layer(bgr_image, 3, 64, "conv1_1")
        self.conv1_2 = self.conv_layer(self.conv1_1, 64, 64, "conv1_2")
        self.pool1 = self.max_pool(self.conv1_2, 'pool1')

        self.conv2_1 = self.conv_layer(self.pool1, 64, 128, "conv2_1")
        self.conv2_2 = self.conv_layer(self.conv2_1, 128, 128, "conv2_2")
        self.pool2 = self.max_pool(self.conv2_2, 'pool2')

        self.conv3_1 = self.conv_layer(self.pool2, 128, 256, "conv3_1")
        self.conv3_2 = self.conv_layer(self.conv3_1, 256, 256, "conv3_2")
        self.conv3_3 = self.conv_layer(self.conv3_2, 256, 256, "conv3_3")
        self.conv3_4 = self.conv_layer(self.conv3_3, 256, 256, "conv3_4")
        self.pool3 = self.max_pool(self.conv3_4, 'pool3')

        self.conv4_1 = self.conv_layer(self.pool3, 256, 512, "conv4_1")
        self.conv4_2 = self.conv_layer(self.conv4_1, 512, 512, "conv4_2")
        self.conv4_3 = self.conv_layer(self.conv4_2, 512, 512, "conv4_3")
        self.conv4_4 = self.conv_layer(self.conv4_3, 512, 512, "conv4_4")
        self.pool4 = self.max_pool(self.conv4_4, 'pool4')

        self.conv5_1 = self.conv_layer(self.pool4, 512, 512, "conv5_1")
        self.conv5_2 = self.conv_layer(self.conv5_1, 512, 512, "conv5_2")
        self.conv5_3 = self.conv_layer(self.conv5_2, 512, 512, "conv5_3")
        self.conv5_4 = self.conv_layer(self.conv5_3, 512, 512, "conv5_4")
        self.pool5 = self.max_pool(self.conv5_4, 'pool5')

        if include_top:
            self.fc6 = self.fc_layer(self.pool5, 25088, 4096, "fc6")  # 25088 = ((224 // (2 ** 5)) ** 2) * 512
            self.relu6 = tf.nn.relu(self.fc6)
            if train_mode is not None:
                self.relu6 = tf.cond(train_mode, lambda: tf.nn.dropout(self.relu6, self.dropout), lambda: self.relu6)
            elif self.trainable:
                self.relu6 = tf.nn.dropout(self.relu6, self.dropout)

            self.fc7 = self.fc_layer(self.relu6, 4096, 4096, "fc7")
            self.relu7 = tf.nn.relu(self.fc7)
            if train_mode is not None:
                self.relu7 = tf.cond(train_mode, lambda: tf.nn.dropout(self.relu7, self.dropout), lambda: self.relu7)
            elif self.trainable:
                self.relu7 = tf.nn.dropout(self.relu7, self.dropout)

            self.fc8 = self.fc_layer(self.relu7, 4096, 1000, "fc8")

            self.prob = tf.nn.softmax(self.fc8, name="prob")

        self.data_dict = None

        return None

    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, in_channels, out_channels, name):
        with tf.variable_scope(name):
            filt, conv_biases = self.get_conv_var(3, in_channels, out_channels, name)

            return tf.nn.relu(tf.nn.bias_add(
                tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME'), conv_biases))

    def fc_layer(self, bottom, in_size, out_size, name):
        with tf.variable_scope(name):
            weights, biases = self.get_fc_var(in_size, out_size, name)

            return tf.nn.bias_add(tf.matmul(tf.reshape(bottom, [-1, in_size]), weights), biases)

    def get_conv_var(self, filter_size, in_channels, out_channels, name):
        initial_value = tf.truncated_normal([filter_size, filter_size, in_channels, out_channels], 0.0, 0.1)
        filters = self.get_var(initial_value, name, 0, name + "_filters")

        initial_value = tf.truncated_normal([out_channels], 0.0, 0.1)
        biases = self.get_var(initial_value, name, 1, name + "_biases")

        return filters, biases

    def get_fc_var(self, in_size, out_size, name):
        initial_value = tf.truncated_normal([in_size, out_size], 0.0, 0.1)
        weights = self.get_var(initial_value, name, 0, name + "_weights")

        initial_value = tf.truncated_normal([out_size], 0.0, 0.1)
        biases = self.get_var(initial_value, name, 1, name + "_biases")

        return weights, biases

    def get_var(self, initial_value, name, idx, var_name):
        if self.data_dict is not None and name in self.data_dict:
            value = self.data_dict[name][idx]
        else:
            value = initial_value

        if self.trainable:
            var = tf.Variable(value, name=var_name)
        else:
            # var = tf.constant(value, dtype=tf.float32, name=var_name)
            var = value

        self.var_dict[(name, idx)] = var

        # print var_name, var.get_shape().as_list()
        # assert var.get_shape() == initial_value.get_shape()

        return var

    def save_npy(self, sess, npy_path="./vgg19-save.npy"):
        assert isinstance(sess, tf.Session)

        data_dict = {}

        for (name, idx), var in list(self.var_dict.items()):
            var_out = sess.run(var)
            if name not in data_dict:
                data_dict[name] = {}
            data_dict[name][idx] = var_out

        np.save(npy_path, data_dict)
        print(("file saved", npy_path))
        return npy_path

    def get_var_count(self):
        count = 0
        for v in list(self.var_dict.values()):
            count += reduce(lambda x, y: x * y, v.get_shape().as_list())
        return count

    def Get_label(self, attribute, wordvec, label_attri, label_WV):
        """
        Get the labels according to attributes.
        :param attribute: The attributes, shape = [batch_size, attribute_length].
        :param wordvec: The word vector, shape = [batch_size, word_vector_length].
        :param label_attri: The list of attributes (np.array), shape = [num_classes, attribute_length].
        Caution: the attributes must be in the order according to the labels.
        :param label_WV: The list of word vectors (np.array), shape = [num_classes, word_vector_length].
        Caution: the word vectors must be in the order according to the labels.
        :return: The relative labels (one-hot), shape = [batch_size, num_classes].
        """
        with tf.Session() as sess:
            [attribute, wordvec] = sess.run([attribute, wordvec])

        # cosine距离
        attribute_WV = np.hstack((attribute, wordvec))
        label_attri_WV = np.hstack((label_attri, label_WV))

        label_idx = list(np.argmax(np.dot(attribute_WV, label_attri_WV.T), axis=1))

        label = np.zeros([np.shape(attribute_WV)[0], np.shape(label_attri_WV)[0]])

        for instance in range(np.shape(label)[0]):
            label[instance][label_idx[instance]] = 1

        return tf.convert_to_tensor(label)

    def acc_label(self, img_label, predic_label):
        right = 0
        with tf.Session() as sess:
            [img_label, predic_label] = sess.run([img_label, predic_label])

        img_label_ = np.argmax(img_label, axis=1)
        predic_label_ = np.argmax(predic_label, axis=1)

        for instance in range(np.shape(img_label_)[0]):
            if img_label_[instance] == predic_label_[instance]:
                right += 1

        return tf.convert_to_tensor(right/np.shape(img_label_)[0])

    def Get_WV(self, label_train, WV_list_train):
        """
        Get the attributes according to labels of training set.
        :param label_train: Labels of training set, shape = [batch_size, num_classes].
        :param WV_list_train: The list of word vectors (np.array), shape = [num_classes, word_vector_length].
        Caution: the attributes must be in the order according to the labels.
        :return: The attributes.
        """
        with tf.Session() as sess:
            label_train = sess.run(label_train)

        return tf.convert_to_tensor(np.dot(label_train, WV_list_train))




