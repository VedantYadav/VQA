import numpy as np
import embedding
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Merge, Reshape, Dropout

def vis_lstm():
	embedding_matrix = embedding.load()
	embedding_model = Sequential()
	embedding_model.add(Embedding(
		embedding_matrix.shape[0],
		embedding_matrix.shape[1],
		weights = [embedding_matrix],
		trainable = False ))
	
	image_model = Sequential()
	image_model.add(Dense(
		embedding_matrix.shape[1],
		input_dim=4096,
		activation='linear' ))
	image_model.add(Reshape((1,embedding_matrix.shape[1])))
	
	main_model = Sequential()
	main_model.add(Merge(
		[image_model,embedding_model],
		mode = 'concat',		
		concat_axis = 1))
	main_model.add(LSTM(1001))
	main_model.add(Dropout(0.5))
	main_model.add(Dense(1001,activation='softmax'))
	
	return main_model

def vis_lstm_2():
	embedding_matrix = embedding.load()
	embedding_model = Sequential()
	embedding_model.add(Embedding(
		embedding_matrix.shape[0],
		embedding_matrix.shape[1],
		weights = [embedding_matrix],
		trainable = False ))
	
	image_model_1 = Sequential()
	image_model_1.add(Dense(
		embedding_matrix.shape[1],
		input_dim=4096,
		activation='linear' ))
	image_model_1.add(Reshape((1,embedding_matrix.shape[1])))
	
	image_model_2 = Sequential()
	image_model_2.add(Dense(
		embedding_matrix.shape[1],
		input_dim=4096,
		activation='linear' ))
	image_model_2.add(Reshape((1,embedding_matrix.shape[1])))
	
	main_model = Sequential()
	main_model.add(Merge(
		[image_model_1,embedding_model,image_model_2],
		mode = 'concat',
		concat_axis = 1 ))
	main_model.add(LSTM(1001))
	main_model.add(Dropout(0.5))
	main_model.add(Dense(1001,activation='softmax'))
	
	return main_model


	
