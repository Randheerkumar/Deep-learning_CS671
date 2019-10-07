## Task
	Given the image of eyes generate the mask of iris

## Required Libaries

	numpy
	os
	glob
	argparse
	opencv
	sklearn
	tensorflow 

## Arguments for data_preprocess.py

	--dataset_path :path of the input data
	--num_examples :number of trainig data to take out of all the data
	--output_path  :output path of the processed data

## Arguments for main.py

	--train_data_path :inout path of the processed training data	
	--phase': for testing 'test' and for training 'train' default is 'train'
	--test_data_input_path :path of testing data
	--mask_output_path : path of output mask
	--model_path :path to save model

## For training

	python3 main.py --phase=train
## For testing 

	main.py --phase=test	
