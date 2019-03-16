import sys
import datetime

import helper
import modelTrainer
import dataProcessor

# To train data into model, run this:
# python main.py Train train.csv
#
# To generate output from test data, run this:
# python main.py Validate test.csv experimental_result.csv

def main():
	action = sys.argv[1]
	action = action.lower()
	start_time = datetime.datetime.now()
	if(action.__eq__('train') and len(sys.argv) == 3):
		inputFile = sys.argv[2]
		input_np = helper.readCsvToNumpy(inputFile)
		# Training model starts here
		modelTrainer.trainModel(input_np)

		# Calculation of total time taken
		print("Start Time: " + str(start_time))
		print("End Time: " + str(datetime.datetime.now()))
		print("Duration: " + str(datetime.datetime.now() - start_time))
		print("Complete training model")
		# Training model ends here
	elif(action.__eq__('validate') and len(sys.argv) == 4):
		inputFile = sys.argv[2]
		outputFile = sys.argv[3]
		input_np = helper.readCsvToNumpy(inputFile)
		# Process test data starts here
		output_np = dataProcessor.processTestData(input_np)
		print("Complete processing data")
		# Process test data ends here
		helper.saveResultToCsv(output_np, outputFile)

		# Calculation of total time taken
		print("Start Time: " + str(start_time))
		print("End Time: " + str(datetime.datetime.now()))
		print("Duration: " + str(datetime.datetime.now() - start_time))
		print("Attempted to save into csv")
	else:
		print("I am sad to say")
		print(".")
		print(".")
		print(".")
		print(".")
		print("You did not feed me well.")
		print("If you want to train me, you must train me with passion + a file name")
		print("Sample: \"python main.py TRAIN train.csv\"")
		print("If you want me to think for your lazy mind, you need to give me 1 more filename")
		print("For example: \"python main.py VALIDATE test.csv result.csv")

if __name__ == '__main__':
	main()