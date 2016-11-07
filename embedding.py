import numpy as np
import h5py
import pickle

def load():
	path = 'embeddings/embedding_matrix.h5'
	with h5py.File(path,'r') as hf:
		data = hf.get('embedding_matrix')
		embedding_matrix = np.array(data)
	return embedding_matrix

def load_idx():
	path = 'embeddings/word_idx'
	with open(path,'r') as file:
		word_idx = pickle.load(file)
	return word_idx


