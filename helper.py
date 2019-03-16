#import cv
import numpy as np
import json
import nltk
from nltk.corpus import stopwords

__path__ = "Test_Data/"
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def getStopWords():
	return stop_words
	
def readCsvToNumpy(inputFile):
	inputFile = __path__ + inputFile
	try:
		input_np = np.genfromtxt(inputFile, delimiter=',', dtype='unicode_')
	except:
		print("I am so disappointed that you failed giving me the correct file :(")
		exit()
	return input_np

def saveResultToCsv(npArr, outputFile):
	print(npArr.ndim)
	fmt = ''
	if(npArr.ndim == 1):
		fmt = '%s'
	elif(npArr.ndim > 1):
		fmt = '%s'
		for i in range(1, npArr.ndim):
			fmt += ',%s'

	outputFile = __path__ + outputFile
	try:
		np.savetxt(outputFile, npArr, delimiter=",", fmt=fmt)
	except:
		print("My brain is fried, please save me... :(")
		exit()

def readImageToNumpy(imageFile):
	img = []
#	try:
#		img = np.array(cv.imread(imageFile, 1))
#	except:
#		print("Are you trying to read something out loud to me? I see no image here.")
#		
	return img

def readJsonFile(jsonFile):
	jsonFile = __path__ + jsonFile
	try:
		with open(jsonFile) as json_file:
			jsonArr = json.load(json_file)
	except:
		print("Stop frying my brain, I can't handle your pan")
	return jsonArr

def saveToJsonFile(json_dict, jsonFile):
	jsonFile = __path__ + jsonFile
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