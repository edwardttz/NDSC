import numpy as np
import datetime
import math
from tensorflow import contrib

import helper

def processTestData(inputFile, outputFile):
	print("Start processing data")

	input_np = helper.readCsvToNumpy(inputFile)
	output_list = [["itemid", "Category"]]
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
		new_img_np = helper.prepareImg(img_np)

		# Start processing via BOW model
		str_list = item_title.split(' ')
		str_count = len(str_list)
		str_combi = helper.getAllCombiFromString(str_list, 0, 0, [])
		result_dict = processDataBagOfWordsModel(str_combi, 0, model_dict, str_count, {})

		if(len(result_dict) >= 1):
			top_result = int(max(result_dict.items(), key=lambda k: k[1])[0])
			word_accuracy = result_dict[top_result]
		else:
			top_result = -1
		# End of BOW processing

		# Start processing via Image Training model
		if(not new_img_np == []):
			prediction, img_accuracy = predictImgCategory(tf_model, new_img_np)
		else:
			prediction = -1
		# End of Image processing

		if(prediction == -1 and top_result == -1):
			final_result = math.randrange(0, 57)
		elif(prediction == top_result):
			final_result = prediction
		elif(prediction == -1):
			final_result = top_result
		elif(top_result == -1):
			final_result = prediction
		elif(word_accuracy > img_accuracy):
			final_result = top_result
		else:
			final_result = prediction

		output_list += [[item_id, final_result]]
		print(str(round(float(count * 100) /len(input_np), 2)) + "%", end="\r", flush=True)
	
	helper.saveResultToCsv(np.array(output_list), outputFile)

def predictImgCategory(tf_model, img_np):
	features = np.array(helper.readPickle("features.pickle")).reshape(-1, IMG_SIZE, IMG_SIZE, 1) / 255.0
	tfe = contrib.eager
	epoch_accuracy = tfe.metrics.Accuracy()
	
	prediction = model.predict(img_np)
	img_accy = epoch_accuracy(tf.argmax(model(features), axis=1, output_type=tf.int32), y).result()
	return int(prediction[0][0]), img_accy

def processDataBagOfWordsModel(str_combi, x, model_dict, str_count, result_dict):
	total_doc = float(len(model_dict))
	count_of_doc = 0.00
	temp_dict = {}

	if(x >= len(str_combi)):
		return result_dict
	else:
		word = str_combi[x]
		for cat_num in model_dict:
			if(word in model_dict[cat_num]):
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