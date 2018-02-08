import os
import shutil

classes_file = open("classes.txt","r")
classes = []

for line in classes_file:
	line = line.strip().split('\t')
	classes.append(line[1])

print(classes)

traing_ratio = 0.8

# for animal_class in classes:
# 	oringin_path_name = './Data/train/' + animal_class 
# 	target_path_name = './Data/validation/' + animal_class
# 	os.makedirs(target_path_name)
# 	image_list = os.listdir(oringin_path_name)
# 	length = len(image_list)
# 	for i in range(length):
# 		if i > length*traing_ratio:
# 			file_name = '/' + image_list[i]
# 			target_file = open(target_path_name+file_name,'w')
# 			shutil.move(oringin_path_name+file_name, target_path_name+file_name)
# 			# print(oringin_path_name+file_name, target_path_name+file_name)

for animal_class in classes:
	oringin_path_name = './Data/ALL/' + animal_class 
	train_path_name = './Data/train/' + animal_class
	os.makedirs(train_path_name)
	target_path_name = './Data/validation/' + animal_class
	os.makedirs(target_path_name)
	image_list = os.listdir(oringin_path_name)
	length = len(image_list)
	for i in range(length):
		if i % 50 == 0:
			file_name = '/' + image_list[i]
			target_file = open(target_path_name+file_name,'w')
			shutil.move(oringin_path_name+file_name, target_path_name+file_name)
			# print(oringin_path_name+file_name, target_path_name+file_name)
		elif i % 10 == 0:
			file_name = '/' + image_list[i]
			train_file = open(train_path_name+file_name,'w')
			shutil.move(oringin_path_name+file_name, train_path_name+file_name)





