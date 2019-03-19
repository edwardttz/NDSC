import numpy as np
import datetime
import math

import helper

def processTestData(input_np):
	print("Start processing data")	
	output_list = [["itemid","BOW Category", "BOW acc", "TF Category", "Same Result"]]
	count = 0
	model_dict = helper.readJsonFile("bag_of_words_model.json")
	tf_model = helper.loadTFModel("img_tf.model")
	print("Complete loading models")

	# Start of testing data
	for row in input_np:
		count += 1
		# Ignore the header line
		if(count == 1):
			continue
		item_id = row[0]
		item_title = row[1]
		img_np = np.array(helper.readImage(row[2]))
		new_img_np = prepareImg(img_np)

		# Start processing via BOW model
		str_list = item_title.split(' ')
		str_count = len(str_list)
		str_combi = helper.getAllCombiFromString(str_list, 0, 0, [])
		result_dict = processDataBagOfWordsModel(str_combi, 0, model_dict, str_count, {})

		if(len(result_dict) >= 1):
			top_result = int(max(result_dict.items(), key=lambda k: k[1])[0])
		else:
			top_result = -1
		# End of BOW processing

		# Start processing via Image Training model
		if(not new_img_np == []):
			prediction = predictImgCategory(tf_model, new_img_np)
		else:
			prediction = -1
		# End of Image processing

		check = (prediction == top_result)
		output_list += [[item_id, top_result, result_dict[top_result] * 100], prediction, check]
		if(count >= 200):
			break

		print(str(round(float(count * 100) /len(input_np), 2)) + "%")

	output_np = np.array(output_list)
	return output_np

def predictImgCategory(tf_model, img_np):
	prediction = model.predict(img_np)
	return int(prediction[0][0])

def processDataBagOfWordsModel(str_combi, x, model_dict, str_count, result_dict):
	STOP_WORDS = helper.getStopWords()
	total_doc = float(len(model_dict))
	count_of_doc = 0.00
	temp_dict = {}

	if(x >= len(str_combi)):
		return result_dict
	else:
		word = str_combi[x]
		for cat_num in model_dict:
			# Ignoring all stopwords from the model
			if(not word in STOP_WORDS and word in model_dict[cat_num]):
				# Calculation of weight - TF
				count_appeared = float(model_dict[cat_num][word])
				num_of_terms = float(len(model_dict[cat_num]))
				if(cat_num not in temp_dict):
					temp_dict[cat_num] = count_appeared / num_of_terms
				else:
					temp_dict[cat_num] += (count_appeared / num_of_terms)
				count_of_doc += 1.00

		# Calculation of weight - TF-IDF
		# Adding into the result dictionary
		for key in temp_dict:
			temp_dict[key] *= (total_doc / count_of_doc)
			if(key not in result_dict):
				result_dict[key] = temp_dict[key]
			else:
				result_dict[key] += temp_dict[key]

		return processDataBagOfWordsModel(str_combi, x+1, model_dict, str_count, result_dict)