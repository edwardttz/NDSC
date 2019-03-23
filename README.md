Install and update the following 3 python packages
1) Tensorflow
2) Numpy
3) opencv
4) pickle
5) nltk

You need to download all the images into the main folders:
1) ./beauty_image
2) ./fashion_image
3) ./mobile_image

Before running the py files, you will need to extract the zip file from ./Test_Data

Once you have done all the agove steps, do the following actions in sequence:
1) python main.py Train train.csv 2
2) python main.py Train train.csv 1
3) python main.py Train train.csv 0
4) python main.py Validate test.csv experimental_result.csv
