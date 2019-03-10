import numpy as np
import datetime

import helper

def trainModel(input_np):
	print("Start training model")
	jsonList = helper.readJsonFile("categories.json")
	# Set up dictionary for respective categories
	model_dict = {}
	for cat_name in jsonList:
		cat_list = jsonList[cat_name]
		for cat_key in cat_list:
			model_dict.update({str(cat_list[cat_key]) : {}})
	count = 0

	# Start of training data
	for row in input_np:
		# Ignore the header line
		count += 1
		if(count == 1):
			continue
		item_title = row[1]
		item_cat = row[2]
		img_np = np.array(helper.readImageToNumpy(row[3]))
		
		# Start of bag-of-words training
		str_list = item_title.split(' ')
		str_combi = helper.getAllCombiFromString(str_list, 0, 0, [])
		#print(str_combi)
		model_dict = trainBagOfWordsModel(str_combi, 0, model_dict, item_cat)
		print(str(round(count/len(input_np) * 100, 2)) + "%")

	# End result of model
	helper.saveToJsonFile(model_dict, "bag_of_words_model.json")

def trainBagOfWordsModel(str_combi, x, model_dict, cat_num):
	if(x >= len(str_combi)):
		return model_dict
	else:
		word = str_combi[x]
		if(word in model_dict[cat_num]):
			model_dict[cat_num][word] += 1
		else:
			model_dict[cat_num][word] = 1
		return trainBagOfWordsModel(str_combi, x+1, model_dict, cat_num)