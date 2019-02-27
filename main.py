import sys
import helper

def main():
	action = sys.argv[1]
	if(action.__eq__('Train')):
		inputFile = sys.argv[2]
		input_np = helper.readCsvToNumpy(inputFile)
		# Training model starts here
		helper.trainModel(input_np)
		# Training model ends here		

	elif(action.__eq__('Validate')):
		inputFile = sys.argv[2]
		outputFile = sys.argv[3]
		input_np = helper.readCsvToNumpy(inputFile)
		# Process test data starts here
		output_np = helper.processTestData(input_np)
		# Process test data ends here
		helper.saveResultToCsv(output_np, outputFile)	

if __name__ == '__main__':
	main()

# To train data into model, run this:
# python main.py Train ./"Test Data"/train.csv
#
# To generate output from test data, run this:
# python main.py Validate ./"Test Data"/test.csv experimental_result.csv