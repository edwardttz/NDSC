import cv2
import numpy as np
import json
import nltk
from nltk.corpus import stopwords

PATH = "Test_Data/"
nltk.download('stopwords')
STOP_WORDS = set(stopwords.words('english'))

def getStopWords():
	return STOP_WORDS
	
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

def readImageToNumpy(imageFile):
	img = []
	try:
		img = cv2.imread(imageFile, cv2.IMREAD_GRAYSCALE)
	except:
		print("Are you trying to read something out loud to me? I see no image here.")
	return img

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