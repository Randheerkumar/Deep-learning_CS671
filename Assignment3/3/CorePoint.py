
#importing various libraries
import numpy as np
import tensorflow as tf
import glob
import os
import argparse
import cv2

#parser for taking argument
parser = argparse.ArgumentParser()
parser.add_argument('--phase', default='train')
parser.add_argument('--epochs', default=100,type=int)
parser.add_argument('--dataset_path', default='dataset')

#read file
def read_file(path):
	file=open(path,"r")
	for line in file:
		s=line.split(" ")
		x=int(s[0])
		y=int(s[1])
		break

	file.close()
	return x,y

#load the training data
def load_train_data(data_path,groud_truth_path):
	num_samples=0

	shape=(480,320)
	dim=(320,480)

	for file in sorted(os.listdir(data_path)):
		if num_samples==0:
			img=cv2.imread(data_path+"/"+file,cv2.IMREAD_GRAYSCALE)
			image_shape=img.shape
		num_samples+=1


	X=np.ndarray(shape=(num_samples,image_shape[0],image_shape[1]), dtype=np.uint8)
	Y=np.ndarray(shape=(num_samples,2), dtype=np.uint8)

	print("starts loading data")
	i=0
	for file in sorted(os.listdir(data_path)):
		img=cv2.imread(data_path+"/"+file,cv2.IMREAD_GRAYSCALE)
		im=np.ndarray(shape=(image_shape[0],image_shape[1],3))

		if i <1000:
			s=file.split(".")
			ground_path=groud_truth_path+"/"+str(s[0])+"_gt.txt"
			x,y=read_file(ground_path)

			if img.shape==shape:
				X[i]=img
				Y[i]=(x,y)


				#im[int(x)][int(y)][0]=255
				im[int(x)-10:int(x)-10,int(y)-10:int(y)+10,0]=255
				cv2.imwrite("/home/dilip/Documents/course/6thsem/DL/Assignment/Assignment3/q3/output/"+str(i)+".png",im)
			else:
				x_r=dim[1]/img.shape[0]
				y_r=dim[0]/img.shape[1]
				img1=cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

				x=x*x_r
				y=y*y_r
				X[i]=img1
				Y[i]=(x,y)

		i+=1
	print("loading completed")
	return X,Y

def load_test_data(data_path):
	num_samples=0

	shape=(480,320)
	dim=(320,480)

	for file in sorted(os.listdir(data_path)):
		num_samples+=1

	X=np.ndarray(shape=(num_samples,shape[0],shape[1]), dtype=np.uint8)
	Y=[]

	print("starts loading test data")
	i=0
	for file in sorted(os.listdir(data_path)):
		img=cv2.imread(data_path+"/"+file,cv2.IMREAD_GRAYSCALE)
		s=file.split(".")
		Y.append(str(s[0])+"_predicted_point"+".txt")

		if img.shape==shape:
			X[i]=img
		else:
			img1=cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
			X[i]=img1
		i+=1
	print("test data loading completed")
	return X,Y

def next_batch(x,y,i,batch_size):
	n=x.shape[0]
	if(i+batch_size<n):
	    x_=x[i:i+batch_size,:,:]
	    y_=y[i:i+batch_size,:]

	else:
	    x_=x[i:,:,:]
	    y_=y[i:,:]

	return x_,y_,i+batch_size 

#this function give th next batch of the testing data
def next_test_batch(x,y,i,batch_size):
	n=x.shape[0]
	if(i+batch_size<n):
		x_=x[i:i+batch_size,:,:]
		y_=y[i:i+batch_size]
	else:
	    x_=x[i:,:,:]
	    y_=y[i:]

	return x_,y_,i+batch_size 

def save_point(output_path,x,y):
	n=x.shape[0]
	i=0
	while i<n:
		path=output_path+y[i]
		val=str(int(x[i][0]))+" "+str(int(x[i][1]))+"\n"
		file = open(path,"w")
		file.write(val)
		file.close()
		i+=1 
  

# Create the neural network
def conv_net(x, n_classes, dropout, reuse, is_training):
    
	# Define a scope for reusing the variables
	# with tf.variable_scope('ConvNet', reuse=reuse):

	x = tf.reshape(x, shape=[-1, 480, 320, 1])

	conv1 = tf.layers.conv2d(inputs=x, filters=16, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
	# Now 480x320x16
	maxpool1 = tf.layers.max_pooling2d(conv1, pool_size=(2,2), strides=(2,2), padding='same')
	# Now 240x160x16
	conv2 = tf.layers.conv2d(inputs=maxpool1, filters=8, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
	# Now 240x160x8
	maxpool2 = tf.layers.max_pooling2d(conv2, pool_size=(2,2), strides=(2,2), padding='same')
	# Now 120x80x8
	conv3 = tf.layers.conv2d(inputs=maxpool2, filters=8, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
	# Now 120x80x8
	maxpool3 = tf.layers.max_pooling2d(conv3, pool_size=(2,2), strides=(2,2), padding='same')
	#Now 60x40x8
	conv4= tf.layers.conv2d(inputs=maxpool3, filters=4, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
	# Now 60x40x4
	maxpool4 = tf.layers.max_pooling2d(conv4, pool_size=(2,2), strides=(2,2), padding='same')
	# Now 30x20x4


	# Flatten the data to a 1-D vector for the fully connected layer
	fc1 = tf.contrib.layers.flatten(maxpool4)

	# Fully connected layer (in tf contrib folder for now)
	fc1 = tf.layers.dense(fc1, 1024)
	# Apply Dropout (if is_training is False, dropout is not applied)
	fc1 = tf.layers.dropout(fc1, rate=dropout, training=is_training)

	# Output layer, class prediction
	out = tf.layers.dense(fc1, n_classes)

	return out




#training function
def train(args):
	dataset= input("Enter the training folder")
	data_path=dataset+"/"+"Data"
	groud_truth_path=dataset+"/"+"Ground_truth"

	x,y=load_train_data(data_path,groud_truth_path)

	num_classes=2
	learning_rate = 0.01
	epochs=args.epochs
	batch_size =100
	# Input and target placeholders
	X= tf.placeholder(tf.float32, (None, 480,320), name="input")
	Y= tf.placeholder(tf.float32, (None, 2), name="target")

	out=conv_net(X,num_classes,0.5,reuse=False, is_training=True)
	#loss=tf.losses.mean_squared_error(labels=Y,predictions=out)
	loss=tf.reduce_mean(tf.math.sqrt(tf.reduce_sum(tf.square(Y-out),1,keepdims=True)))
	opt = tf.train.AdamOptimizer(learning_rate).minimize(loss)

	init=tf.global_variables_initializer()

	#saver_path="/home/dilip/Documents/course/6thsem/DL/Assignment/Assignment3/q3/"
	saver_path="tmp/"
	with tf.Session() as sess:
		sess.run(init)
		n=x.shape[0]
		for e in range(100):
			saver = tf.train.Saver()
			i=0;batch_num=0
			while i<n:
				x_,y_,i=next_batch(x,y,i,batch_size)
				batch_loss,_,y_pred = sess.run([loss, opt,out], feed_dict={X: x_,Y: y_})
				batch_num+=1
				print("epoch=",e,"batch_num=",batch_num,"loss=",batch_loss,"actual=",y_[0],"predicted=",y_pred[0])
			save_path = saver.save(sess, saver_path+"model"+str(e+1))
			print("Model saved in path: %s" % save_path)




def test(args):
	data_path= input("Enter the testing folder")
	x,y=load_test_data(data_path)
	output_path="output/"

	num_classes=2
	learning_rate = 0.001
	epochs=args.epochs
	batch_size =100
	n_test=x.shape[0]
	# Input and target placeholders
	X= tf.placeholder(tf.float32, (None, 480,320), name="input")
	# #Y= tf.placeholder(tf.float32, (None, 2), name="target")
	out=conv_net(X,num_classes,0.5,reuse=True, is_training=False)
	#tf.reset_default_graph()

	# import the graph from the file
	saver_path="tmp1/"
	#imported_graph = tf.train.import_meta_graph(saver_path+"model1.meta")

	saver = tf.train.Saver()

	#saver_path="/home/dilip/Documents/course/6thsem/DL/Assignment/Assignment3/q3/tmp/"
	with tf.Session() as sess:  
		#saver = tf.train.import_meta_graph(saver_path+"model1")
		#saver.restore(sess,tf.train.latest_checkpoint(saver_path+"./"))
		saver.restore(sess, saver_path+"model6")
		i=0
		while i<n_test:
			x1,y1,i=next_test_batch(x,y,i,batch_size)
			x_=sess.run(out, feed_dict={X: x1})
			save_point(output_path,x_,y1)




def main(args):
	if args.phase=="train":
		train(args)
	else:
		test(args)



#parsing argument and calling main function
args = parser.parse_args()
main(args)	



