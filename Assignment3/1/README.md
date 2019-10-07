
# object detection and localization

		part1 : classification  and localization of object in a imgage

		part2 : localization of four object in a image


# step involved 

	1 . scaling of images so that size of all images would be same

	2 . preprocessing of images into numpy array

	3 . classification and regression:
			feature extracting, multihead classification

	4.testing

	5 . IUO calculation







#  method for localization single obejct and its classification

## using multihead classification

	    classification head : cross entropy loss

	    regression head  : mean squared loss

## for part 1

	python task1.py

## for part 2

	python task2.py


# dependencies:
		tensorflow

		matplotlib

		numpy

		scikit-learn	



# result  :
		classification accuracy :100%

		IOU : result1.txt  (IOU : intersection over union between ground trouth and prediction)
                almost 85% images have IOU value greater than 0.5







