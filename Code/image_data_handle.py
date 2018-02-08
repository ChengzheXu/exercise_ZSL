from skimage import io, transform
from PIL import Image
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle

img_width = 224
img_height = 224
pickle_load_file = open('./Data/pickledata/train_and_vali_file_list.pkl', 'rb')
temp_list = pickle.load(pickle_load_file)
train_file_list, vali_file_list = temp_list[0], temp_list[1]
train_file_len, vali_file_len = len(train_file_list), len(vali_file_list)
print(train_file_len,vali_file_len)
classes_list = temp_list[2]
classes_new_list = temp_list[3]
classes_new_dict = {item:index for index, item in enumerate(classes_new_list)}
attribute_list = temp_list[4]
attribute_bi_dict = temp_list[5]
attribute_conti_dict = temp_list[6]

attribute_list_all = np.zeros((25,85))

for index, each_class in enumerate(classes_new_list):
	attribute_list_all[index,:] = attribute_bi_dict[each_class]

attribute_list_all = attribute_list_all.astype('float32')

word2vec_load_file = open('./glove/word2vec_pickle.pkl','rb')
word2vec_list = pickle.load(word2vec_load_file)
word2vec_50 = word2vec_list[0]
word2vec_100 = word2vec_list[1]
word2vec_200 = word2vec_list[2]
word2vec_300 = word2vec_list[3]

def image2tensor(image_path_name):
	img = mpimg.imread(image_path_name)
	new_image = transform.resize(img, (img_width, img_height))
	# new_image_ = np.asarray(new_image)
	# new_image = tf.convert_to_tensor(new_image)
	return new_image

def glove(word, vec_length):
	if vec_length == 50:
		try:
			word2vec = np.array(word2vec_50[word])
		except KeyError:
			word2vec = np.zeros(50)
	elif vec_length == 100:
		try:
			word2vec = np.array(word2vec_100[word])
		except KeyError:
			word2vec = np.zeros(100)
	elif vec_length == 200:
		try:
			word2vec = np.array(word2vec_200[word])
		except KeyError:
			word2vec = np.zeros(200)
	elif vec_length == 300:
		try:
			word2vec = np.array(word2vec_300[word])
		except KeyError:
			word2vec = np.zeros(300)
	else:
		print('vec_length should be 50, 100, 200, or 300！')
		return 0
	return word2vec



def Get_next_batch(Train_or_Vali, batch_size, epoch, use_word2vec=0, attribute = 'conti'):
	"""
	Get next batch according to size.
	:param Train_or_Vali: Whether to train or validate the model.
	:param batch_size: The size of the training batch.
	:param use_word2vec: If use the word2vec feature as an attribute
	:return: training data, shape = [batch_size, img_width, img_height, img_path],
	attribute, shape = [batch_size, attribute_length]
	and label(one-hot), shape = [batch_size, num_classes]
	for next batch, type = np.array.
	"""
	img_batch_tensor = np.zeros((batch_size, img_width, img_height, 3))
	if not use_word2vec:
		attribute_batch_tensor = np.ndarray((batch_size, 85))
	else:
		attribute_batch_tensor = np.ndarray((batch_size, 85 + use_word2vec))
	# if not use_all_label:
	# 	label_batch_tensor = np.ndarray((batch_size, 20))
	# else:
	label_batch_tensor = np.zeros((batch_size, 25))


	if Train_or_Vali == 'train':
		img_path = './Data/train_zsl/'
		for batch in range(batch_size):
			file_name = train_file_list[(batch_size*epoch + batch)%train_file_len]
			img_batch_tensor[batch] = image2tensor(img_path + file_name)
			label_name = file_name.split('_')[0]
			if attribute == 'bi':
				if not use_word2vec:
					attribute_batch_tensor[batch] = attribute_bi_dict[label_name]
				else:
					attribute_batch_tensor[batch][:85] = attribute_bi_dict[label_name]
					attribute_batch_tensor[batch][85:] = glove(label_name,use_word2vec)
			elif attribute == 'conti':
				if not use_word2vec:
					attribute_batch_tensor[batch] = attribute_conti_dict[label_name]
				else:
					attribute_batch_tensor[batch][:85] = attribute_conti_dict[label_name]
					attribute_batch_tensor[batch][85:] = glove(label_name,use_word2vec)
			label_batch_tensor[batch][classes_new_dict[label_name]] = 1 

	elif Train_or_Vali == 'validation':
		img_path = './Data/validation_zsl/'
		for batch in range(batch_size):
			file_name = vali_file_list[(batch_size*epoch + batch)%vali_file_len]
			img_batch_tensor[batch] = image2tensor(img_path + file_name)
			label_name = file_name.split('_')[0]
			if attribute == 'bi':
				if not use_word2vec:
					attribute_batch_tensor[batch] = attribute_bi_dict[label_name]
				else:
					attribute_batch_tensor[batch][:85] = attribute_bi_dict[label_name]
					attribute_batch_tensor[batch][85:] = glove(label_name,use_word2vec)
			elif attribute == 'conti':
				if not use_word2vec:
					attribute_batch_tensor[batch] = attribute_conti_dict[label_name]
				else:
					attribute_batch_tensor[batch][:85] = attribute_conti_dict[label_name]
					attribute_batch_tensor[batch][85:] = glove(label_name,use_word2vec)
			label_batch_tensor[batch][classes_new_dict[label_name]] = 1
	else:
		print("ERROR!You should input train or validation")

	return img_batch_tensor.astype('float32'), attribute_batch_tensor.astype('float32'), label_batch_tensor.astype('float32')

def main():
	print('test:')
	print('*' * 30)
	image_name = 'pic1.jpg'
	result = image2tensor(image_name)
	Get_next_batch('train', 2, 1)

if __name__ == '__main__':
	main()
