import sys
import helper

def main():
	action = sys.argv[1]
	if(action.__eq__('Train')):
		inputFile = sys.argv[2]
		input_np = helper.readCsvToNumpy(inputFile)
		# Training

	elif(action.__eq__('Validate')):
		inputFile = sys.argv[2]
		outputFile = sys.argv[3]
		input_np = helper.readCsvToNumpy(inputFile)
		# Process inputs
		helper.saveResultToCsv(input_np, outputFile)	

if __name__ == '__main__':
	main()

#filename = experimental_result.csv