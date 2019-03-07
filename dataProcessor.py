import numpy as np

def processTestData(input_np):
	output_np = np.array(['itemid','Category'])
	count = 0
	for row in input_np:
		if(count == 0):
			count += 1
			continue
		item_id = row[0]
		item_title = row[1]
		img_np = np.array(helper.readImageToNumpy(row[2]))
	
	return output_np