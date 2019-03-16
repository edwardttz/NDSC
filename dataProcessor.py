import numpy as np
import datetime
import math

import helper

def processTestData(input_np):
	print("Start processing data")
	model_dict = helper.readJsonFile("bag_of_words_model.json")
	output_list = [['itemid','Category']]
	count = 0

	for row in input_np:
		count += 1
		if(count == 1):
			continue
		item_id = row[0]
		item_title = row[1]
		img_np = np.array(helper.readImageToNumpy(row[2]))
		
		# Start processing via BOW model
		str_list = item_title.split(' ')
		str_count = len(str_list)
		str_combi = helper.getAllCombiFromString(str_list, 0, 0, [])
		result_dict = processDataBagOfWordsModel(str_combi, 0, model_dict, str_count, {})

		if(len(result_dict) >= 1):
			top_result = max(result_dict.items(), key=lambda k: k[1])[0]
		else:
			top_result = -1
		#if(count >= 50):
		#	break
		output_list += [[item_id, top_result]]
		# End of BOW processing

		# Start processing via Image Training model



		# End of Image processing


		print(str(round(count/len(input_np) * 100, 2)) + "%")

	output_np = np.array(output_list)
	return output_np

def processDataBagOfWordsModel(str_combi, x, model_dict, str_count, result_dict):
	stop_words = helper.getStopWords()
	total_doc = len(model_dict)
	tf = 0
	count_of_doc = 0
	temp_dict = {}

	if(x >= len(str_combi)):
		return result_dict
	else:
		word = str_combi[x]
		for cat_num in model_dict:
			# Ignoring all stopwords from the model
			if(not word in stop_words and word in model_dict[cat_num]):
				# Calculation of weight - TF
				count_appeared = model_dict[cat_num][word]
				num_of_terms = len(model_dict[cat_num])
				if(cat_num not in temp_dict):
					temp_dict[cat_num] = count_appeared / num_of_terms
				else:
					temp_dict[cat_num] += (count_appeared / num_of_terms)
				count_of_doc += 1

		# Calculation of weight - TF-IDF
		# Adding into the result dictionary
		for key in temp_dict:
			temp_dict[key] *= (total_doc / count_of_doc)
			if(key not in result_dict):
				result_dict[key] = temp_dict[key]
			else:
				result_dict[key] += temp_dict[key]

		return processDataBagOfWordsModel(str_combi, x+1, model_dict, str_count, result_dict)