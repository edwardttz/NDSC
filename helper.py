import cv2
import numpy as np
import json

__path__ = "Test_Data/"

def readCsvToNumpy(inputFile):
	inputFile = __path__ + inputFile
	try:
		input_np = np.genfromtxt(inputFile, delimiter=',', dtype='unicode_')
	except:
		print("I am so disappointed that you failed giving me the correct file :(")
		exit()
	return input_np

def saveResultToCsv(npArr, outputFile):
	fmt = ''
	if(npArr.shape[0] != 0 and npArr.shape[1] >= 1):
		fmt = '%s'
		for i in range(npArr.shape[1] - 1):
			fmt += ',%s'
		outputFile = __path__ + outputFile
		try:
			np.savetxt(outputFile, npArr, delimiter=",", fmt=fmt)
		except:
			print("My brain is fried, please save me... :(")
			exit()

def readImageToNumpy(imageFile):
	try:
		img = np.array(cv2.imread(imageFile, 1))
	except:
		print("Are you trying to read something out loud to me? I see no image here.")
		img = []
	return img

def readJsonFile(jsonFile):
	jsonFile = __path__ + jsonFile
	try:
		with open(jsonFile) as json_file:
			jsonArr = json.load(json_file)
	except:
		print("Stop frying my brain, I can't handle your pan")
	return jsonArr