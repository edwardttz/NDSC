import numpy as np
import helper

def trainModel(input_np):
	jsonList = helper.readJsonFile("categories.json")
	print(jsonList)
	print(helper.readImageToNumpy("mobile_image/000a5df2a604db41dd082527ea71b4b6.jpg"))