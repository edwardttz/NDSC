import numpy as np
import helper

def trainModel(input_np):
	jsonList = helper.readJsonFile("categories.json")
	print(jsonList)
	count = 0
	for row in input_np:
		# Ignore the header line
		if(count == 0):
			count += 1
			continue
		item_id = row[0]
		item_title = row[1]
		item_cat = row[2]
		img_np = np.array(helper.readImageToNumpy(row[3]))
		print(img_np)
		break