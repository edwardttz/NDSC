import numpy as np
import datetime

import helper

def trainModel(input_np):
	print("Start training model")
	jsonList = helper.readJsonFile("categories.json")

	# Set up dictionary for respective categories
	bow_model_dict = {}
	for cat_name in jsonList:
		cat_list = jsonList[cat_name]
		for cat_key in cat_list:
			bow_model_dict.update({str(cat_list[cat_key]) : {}})
	count = 0

	# Start of training data
	for row in input_np:
		# Ignore the header line
		count += 1
		if(count == 1):
			continue
		item_title = row[1]
		item_cat = row[2]
		#img_np = np.array(helper.readImageToNumpy(row[3]))
		
		# Start of BOW training
		str_list = item_title.split(' ')
		str_combi = helper.getAllCombiFromString(str_list, 0, 0, [])
		#print(str_combi)
		bow_model_dict = trainBagOfWordsModel(str_combi, 0, bow_model_dict, item_cat)
		# Completion of BOW

		# Start of Image training


		# Completion of Image training
		
		print(str(round(float(count * 100) /len(input_np), 2)) + "%")

	# End result of model
	helper.saveToJsonFile(bow_model_dict, "bag_of_words_model.json")

def trainBagOfWordsModel(str_combi, x, model_dict, cat_num):
	stop_words = helper.getStopWords()

	if(x >= len(str_combi)):
		return model_dict
	else:
		word = str_combi[x]
		# Ignoring all stopwords and numbers only from the model
		if(not word in stop_words and not word.isdigit()):
			if(word in model_dict[cat_num]):
				model_dict[cat_num][word] += 1
			else:
				model_dict[cat_num][word] = 1
		return trainBagOfWordsModel(str_combi, x+1, model_dict, cat_num)