import pickle

def handle(file_name):
	word2vec_dict = {}
	file = open(file_name,'r')
	for line in file:
		line = line.strip().split()
		word = line[0]
		line = line[1:]
		vector = [float(i) for i in line]
		word2vec_dict[word] = vector
	return word2vec_dict

file_50_name = './glove/glove.6B.50d.txt'
file_100_name = './glove/glove.6B.100d.txt'
file_200_name = './glove/glove.6B.200d.txt'
file_300_name = './glove/glove.6B.300d.txt'

word2vecs_50 = handle(file_50_name)
word2vecs_100 = handle(file_100_name)
word2vecs_200 = handle(file_200_name)
word2vecs_300 = handle(file_300_name)

word2vec_picklefile = open('./glove/word2vec_pickle.pkl','wb')
pickle.dump([word2vecs_50,word2vecs_100,word2vecs_200,word2vecs_300], word2vec_picklefile)