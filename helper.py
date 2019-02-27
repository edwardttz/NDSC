import csv
import numpy as np

def readCsvToNumpy(inputFile):
	input_np = np.genfromtxt(inputFile, delimiter=',', dtype='unicode_')
	return input_np

def saveResultToCsv(npArr, outputFile):
	fmt = '%s,%s'
	np.savetxt(outputFile, npArr, delimiter=",", fmt=fmt)

def trainModel(input_np):



def processTestData(input_np):