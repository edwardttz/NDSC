import sys
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
	if(action.__eq__('Train') and len(sys.argv) == 3):
		inputFile = sys.argv[2]
		input_np = helper.readCsvToNumpy(inputFile)
		# Training model starts here
		print("Start training model")
		modelTrainer.trainModel(input_np)
		print("Complete training model")
		# Training model ends here
	elif(action.__eq__('Validate') and len(sys.argv) == 4):
		inputFile = sys.argv[2]
		outputFile = sys.argv[3]
		input_np = helper.readCsvToNumpy(inputFile)
		# Process test data starts here
		print("Start processing data")
		output_np = dataProcessor.processTestData(input_np)
		print("Complete processing data")
		# Process test data ends here
		helper.saveResultToCsv(output_np, outputFile)	
		print("Saved to csv")
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