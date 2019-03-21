import sys
import datetime

import modelTrainer
import dataProcessor

# To train data into model, run this:
# python main.py Train train.csv <number>
# 0: Train TF Model; 1: Prepare TF Model; 2: Train BOW Model
#
# To generate output from test data, run this:
# python main.py Validate test.csv experimental_result.csv

def main():
	action = sys.argv[1].lower()
	start_time = datetime.datetime.now()

	if(action.__eq__('train') and len(sys.argv) == 4):
		# Training model starts here
		modelTrainer.trainModel(sys.argv[2], int(sys.argv[3]))

		# Calculation of total time taken
		print("Start Time: " + str(start_time))
		print("End Time: " + str(datetime.datetime.now()))
		print("Duration: " + str(datetime.datetime.now() - start_time))
		print("Complete training model")
		# Training model ends here
	elif(action.__eq__('validate') and len(sys.argv) == 4):
		# Process test data starts here
		dataProcessor.processTestData(sys.argv[2], sys.argv[3])

		# Calculation of total time taken
		print("Start Time: " + str(start_time))
		print("End Time: " + str(datetime.datetime.now()))
		print("Duration: " + str(datetime.datetime.now() - start_time))
		print("Attempted to save into csv")
	else:
		print("I am sad to say")
		print("You did not feed me well.")
		print("If you want to train me, you must train me with passion + a file name")
		print("Sample: \"python main.py TRAIN train.csv 2\"")
		print("If you want me to think for your lazy mind, you need to give me 1 more filename")
		print("For example: \"python main.py VALIDATE test.csv result.csv")

if __name__ == '__main__':
	main()