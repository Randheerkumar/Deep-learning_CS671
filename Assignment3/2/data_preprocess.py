''' data preprocessing '''

#import the required library
import numpy as np
import glob
import os
import argparse
import cv2
from collections import defaultdict
from sklearn.model_selection import train_test_split


#creating the parser for adding arguement 
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_path', default='dataset')
parser.add_argument('--num_examples', default=2000,type=int)
parser.add_argument('--output_path', default='dataset/output')



#load the original images and ground truth mask  and converts to numpy array 
def build_data(mask_path,data_path,num_examples):
	image=defaultdict(list)

	#loading original images
	i=0
	for file in sorted(os.listdir(data_path)):
		img=cv2.imread(data_path+"/"+file)
		if i==0:
			shape_image=img.shape
		break

	i=0
	for file in sorted(os.listdir(mask_path)):
		img=cv2.imread(mask_path+"/"+file)
		if i==0:
			shape_groundtruth=img.shape
		break

	X=np.ndarray(shape=(num_examples,shape_image[0],shape_image[1],shape_image[2]), dtype=np.uint8)
	Y=np.ndarray(shape=(num_examples,shape_groundtruth[0],shape_groundtruth[1],1), dtype=np.uint8)
	
	# X=np.zeros((num_examples,shape_image[0],shape_image[1],shape_image[2]))
	# Y=np.zeros((num_examples,shape_groundtruth[0],shape_groundtruth[1],shape_groundtruth[2]))

	i=0
	for file in sorted(os.listdir(data_path)):
		img=cv2.imread(data_path+"/"+file)
		if i<num_examples:
			X[i]=img
			print("data=",i)
		else:
			break
		i+=1

	#loading groundtruth masks
	i=0
	for file in sorted(os.listdir(mask_path)):
		img=cv2.imread(mask_path+"/"+file)
		if i<num_examples:
			Y[i,:,:,0]=img[:,:,0]
			print("data=",i)
		else:
			break
		i+=1

	return X,Y

#main function of the program
def main(args):
	print("np=",np.intp)
	#various arguenmets
	dataset_path=args.dataset_path
	num_examples=args.num_examples
	output_path=args.output_path

	print("num_exaples=",args.num_examples)

	#path of data and mask
	data_path=dataset_path+"/"+"Data"
	mask_path=dataset_path+"/"+"Mask"

	#x=np.zeros((10000,400,300,3))
	#preprocess data
	X,Y=build_data(mask_path,data_path,num_examples)

	#if output path not exits creates it
	if not os.path.exists(output_path):
		os.makedirs(output_path)

	#split data into train and test
	xTrain, xTest, yTrain, yTest = train_test_split(X, Y, test_size = 0.2, random_state =0) 
	# print(xTrain.shape)	
	print("saving")
	#save the data in .npy format
	# np.save(output_path+"/"+"X",X)
	# np.save(output_path+"/"+"Y",Y)
	np.save(output_path+"/"+"x_train",xTrain)
	np.save(output_path+"/"+"x_test",xTest)
	np.save(output_path+"/"+"y_train",yTrain)
	np.save(output_path+"/"+"y_test",yTest)



#parsing argument and calling main function
args = parser.parse_args()
main(args)	
