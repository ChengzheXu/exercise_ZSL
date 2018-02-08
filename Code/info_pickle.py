import pickle
import os
import random

picklefile = open('./Data/pickledata/train_and_vali_file_list.pkl', 'wb')

train_path = './Data/train_zsl'
train_file_list = os.listdir(train_path)
random.shuffle(train_file_list)

vali_path = './Data/validation_zsl'
vali_file_list = os.listdir(vali_path)
random.shuffle(vali_file_list)

classes_file = open('classes.txt', 'r')
classes_new_file = open('classes_new.txt','r')
attribute_file = open('predicates.txt', 'r')
attribute_bi_file = open('predicate-matrix-binary.txt', 'r')
attribute_conti_file = open('predicate-matrix-continuous.txt', 'r')

classes_list = []
for line in classes_file:
    if not line:
        continue
    line = line.strip().split('\t')
    classes_list.append(line[1])
print(classes_list)

classes_new_list = []
for line in classes_new_file:
    if not line:
        continue
    line = line.strip()
    classes_new_list.append(line)
print(classes_new_list)

attribute_list = []
for line in attribute_file:
    if not line:
        continue
    line = line.strip().split('\t')
    attribute_list.append(line[1])
print(attribute_list)

attribute_bi_dict = {}
index = 0
for line in attribute_bi_file:
    if not line:
        continue
    line = line.strip().split()
    attribute_bi_dict[classes_list[index]] = [float(i) for i in line]
    index += 1
# print(attribute_bi_dict)

attribute_conti_dict = {}
index = 0
for line in attribute_conti_file:
    if not line:
        continue
    line = line.strip().split()
    attribute_conti_dict[classes_list[index]] = [float(i) for i in line]
    index += 1
# print(attribute_conti_dict)

pickle.dump([train_file_list, vali_file_list, classes_list, classes_new_list,
             attribute_list, attribute_bi_dict, attribute_conti_dict], picklefile)
