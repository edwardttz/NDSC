import json
import numpy as np
import cv2
import nltk
from nltk.corpus import stopwords
import tensorflow as tf
import pickle

PATH = "Test_Data/"
IMG_SIZE = 100

def getStopWords():
	nltk.download('stopwords')
	STOP_WORDS = set(stopwords.words('english'))
	return STOP_WORDS

def getImgSize():
	return IMG_SIZE

def readCsvToNumpy(inputFile):
	inputFile = PATH + inputFile
	try:
		input_np = np.genfromtxt(inputFile, delimiter=',', dtype='unicode_')
	except:
		print("I am so disappointed that you failed giving me the correct file :(")
		exit()
	return input_np

def saveResultToCsv(npArr, outputFile):
	fmt = ''
	if(npArr.ndim == 1):
		fmt = '%s'
	elif(npArr.ndim > 1):
		fmt = '%s'
		for i in range(1, npArr.ndim):
			fmt += ',%s'

	outputFile = PATH + outputFile
	try:
		np.savetxt(outputFile, npArr, delimiter=",", fmt=fmt)
	except:
		print("My brain is fried, please save me... :(")
		exit()

def readImage(imageFile):
	img = []
	try:
		img = cv2.imread(imageFile, cv2.IMREAD_GRAYSCALE)
	except:
		print("Are you trying to read something out loud to me? I see no image here.")
	return img

def readPickle(fileName):
	fileName = PATH + fileName
	return pickle.load(open(fileName,"rb"))

def savePickle(fileName, item):
	fileName = PATH + fileName
	pickle_out = open(fileName,"wb")
	pickle.dump(item, pickle_out)
	pickle_out.close()

def readJsonFile(jsonFile):
	jsonFile = PATH + jsonFile
	try:
		with open(jsonFile) as json_file:
			jsonArr = json.load(json_file)
	except:
		print("Stop frying my brain, I can't handle your pan")
	return jsonArr

def saveToJsonFile(json_dict, jsonFile):
	jsonFile = PATH + jsonFile
	with open(jsonFile, "w") as json_write:
		json.dump(json_dict, json_write)

def getAllCombiFromString(str_list, x, y, combi_list):
	if(x >= len(str_list)):
		return combi_list
	elif(y >= len(str_list)):
		return getAllCombiFromString(str_list, x+1, x+1, combi_list)
	else:
		word = str_list[x]
		for i in range(x+1,y+1):
			word += ' ' + str_list[i]
		combi_list.append(word)
		return getAllCombiFromString(str_list, x, y+1, combi_list)

def saveTFModel(model, fileName):
	fileName = PATH + fileName
	model.save(fileName)

def loadTFModel(fileName):
	fileName = PATH + fileName
	model = tf.keras.models.load_model(fileName)
	return model

def prepareImg(img_np):
	new_img_np, valid = resizeImg(img_np)
	if(valid == 0):
		return new_img_np.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
	else:
		return []

def resizeImg(img_np):
	img_arr = []
	try:
		img_arr = cv2.resize(img_np, (IMG_SIZE, IMG_SIZE))
		return img_arr, 0
	except Exception as e:
		return img_arr, -1