import numpy as np
import helper
import datetime

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
		
		# Start processing via bag of word model 
		str_list = item_title.split(' ')
		str_count = len(str_list)
		str_combi = helper.getAllCombiFromString(str_list, 0, 0, [])
		result_dict = processDataBagOfWordsModel(str_combi, 0, model_dict, str_count, {})
		top_result = max(result_dict.items(), key=lambda k: k[1])[0]
		#if(count >= 50):
		#	break
		output_list += [[item_id, top_result]]
		print(str(int(count/len(input_np) * 100)) + "%")


	output_np = np.array(output_list)
	return output_np

def processDataBagOfWordsModel(str_combi, x, model_dict, str_count, result_dict):
	if(x >= len(str_combi)):
		return result_dict
	else:
		word = str_combi[x]
		str_list = word.split(' ')
		weight = len(str_list) / str_count
		for cat_num in model_dict:
			if(word in model_dict[cat_num] and cat_num not in result_dict):
				result_dict[cat_num] = weight
			elif(word in model_dict[cat_num]):
				result_dict[cat_num] += weight
		return processDataBagOfWordsModel(str_combi, x+1, model_dict, str_count, result_dict)
