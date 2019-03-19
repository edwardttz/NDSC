import numpy as np
import datetime
import cv2
import time

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard

import helper

def trainModel(input_np):
	print("Start training model")
	jsonList = helper.readJsonFile("categories.json")
	features = []
	label = []

	# Set up dictionary for respective categories
	bow_model_dict = {}
	for cat_name in jsonList:
		cat_list = jsonList[cat_name]
		for cat_key in cat_list:
			bow_model_dict.update({str(cat_list[cat_key]) : {}})
	count = 0

	# Start of training data
	for row in input_np:
		# Ignore the header line
		count += 1
		if(count == 1):
			continue
		item_title = row[1]
		item_cat = row[2]
		img_np = np.array(helper.readImage(row[3]))

		# Start of BOW training
		str_list = item_title.split(' ')
		str_combi = helper.getAllCombiFromString(str_list, 0, 0, [])
		bow_model_dict = trainBagOfWordsModel(str_combi, 0, bow_model_dict, item_cat)
		# Completion of BOW training

		# Start of Image training part 1 - Extraction of features and label
		features, label = insertRawImageData(features, label, img_np, item_cat)
		# Completion of Image training part 1
		
		print(str(round(float(count * 100) /len(input_np), 2)) + "%")

	# Start of Image training part 2 - Training of model
	dense_layers = [0, 1, 2]
	layer_sizes = [32, 64, 128]
	conv_layers = [1, 2, 3]
	IMG_SIZE = helper.getImgSize()
	features = np.array(features).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
	features = features / 255.0
	model = trainTFModel(dense_layers, layer_sizes, conv_layers, features, label)
	# Completion of Image training part 2

	# End result of model
	helper.saveToJsonFile(bow_model_dict, "bag_of_words_model.json")
	helper.saveTFModel(model, "img_tf.model")

def trainBagOfWordsModel(str_combi, x, model_dict, cat_num):
	STOP_WORDS = helper.getStopWords()

	if(x >= len(str_combi)):
		return model_dict
	else:
		word = str_combi[x]
		# Ignoring all stopwords and numbers only from the model
		if(not word in STOP_WORDS and not word.isdigit()):
			if(word in model_dict[cat_num]):
				model_dict[cat_num][word] += 1
			else:
				model_dict[cat_num][word] = 1
		return trainBagOfWordsModel(str_combi, x+1, model_dict, cat_num)

def insertRawImageData(features, label, img_np, cat_num):
	IMG_SIZE = helper.getImgSize()
	img_arr, valid = helper.resizeImg(img_np)
	if(valid == 0):
		features.append(img_arr)
		label.append(cat_num)

	return features, label

def trainTFModel(dense_layers, layer_sizes, conv_layers, X, y):
	for dense_layer in dense_layers:
		for layer_size in layer_sizes:
			for conv_layer in conv_layers:
				NAME = "{}-conv-{}-nodes-{}-dense-{}".format(conv_layer, layer_size, dense_layer, int(time.time()))
				tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))
				model = Sequential()

				model.add(Conv2D(layer_size, (3, 3), input_shape=X.shape[1:]))
				model.add(Activation('relu'))
				model.add(MaxPooling2D(pool_size=(2, 2)))

				for l in range(conv_layer - 1):
					model.add(Conv2D(layer_size, (3, 3)))
					model.add(Activation('relu'))
					model.add(MaxPooling2D(pool_size=(2, 2)))

				model.add(Flatten())

				for _ in range(dense_layer):
					model.add(Dense(layer_size))
					model.add(Activation('relu'))

				model.add(Dense(1))
				model.add(Activation('sigmoid'))

				model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
				model.fit(X, y, batch_size=32, epochs=10, validation_split=0.3, callbacks=[tensorboard])

	return model