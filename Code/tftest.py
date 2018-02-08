import tensorflow as tf
import numpy as np

test = np.ndarray((3,224,224,3))
img_tensor = tf.placeholder(tf.float32, shape=[None, 224, 224, 3])
