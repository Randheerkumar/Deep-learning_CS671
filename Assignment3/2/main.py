
#importing required libraries
import numpy as np
import tensorflow as tf
import argparse
import cv2
import os


#parser for taking argument
parser = argparse.ArgumentParser()
parser.add_argument('--phase', default='train')
parser.add_argument('--train_data_path', default='dataset/output1/')
parser.add_argument('--test_data_input_path', default='dataset/test_data/')
parser.add_argument('--mask_output_path', default='predicted_mask/')
parser.add_argument('--model_path', default='tmp/')


#this function give th next batch of the training data
def next_batch(x,y,i,batch_size):
    n=x.shape[0]
    if(i+batch_size<n):
        x_=x[i:i+batch_size,:,:,:]
        y_=y[i:i+batch_size,:,:,:]

    else:
        x_=x[i:,:,:,:]
        y_=y[i:,:,:,:]
    
    return x_,y_,i+batch_size 

 
#this function give th next batch of the testing data
def next_test_batch(x,y,i,batch_size):
	n=x.shape[0]
	if(i+batch_size<n):
		x_=x[i:i+batch_size,:,:,:]
		y_=y[i:i+batch_size]
	else:
	    x_=x[i:,:,:,:]
	    y_=y[i:]

	return x_,y_,i+batch_size 


#load the test images and corresponding image name
def load_test_data(data_path):
	num_examples=0

	#counting number of test data
	i=0
	for file in sorted(os.listdir(data_path)):
		num_examples+=1
		if i==0:
			img=cv2.imread(data_path+"/"+file)
			shape_image=img.shape
		i+=1

	X=np.ndarray(shape=(num_examples,shape_image[0],shape_image[1],shape_image[2]), dtype=np.uint8)
	Y=[]
	i=0
	for file in sorted(os.listdir(data_path)):
		img=cv2.imread(data_path+"/"+file)
		X[i]=img
		s=file.split(".")
		Y.append(str(s[0])+"_predicted_mask"+".png")
		print("data=",i)
		i+=1

	return X,Y

#this function save the predicted mask of test data in the given folder	
def save_predicted_mask(path,x_,y_path):
	n=x_.shape[0]
	for i in range(n):
		file_path=path+y_path[i]
		cv2.imwrite(file_path,x_[i])

### Encoder
def encoder(inputs):
	conv1 = tf.layers.conv2d(inputs=inputs, filters=16, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
	# Now 300x400x16
	conv1 =tf.layers.batch_normalization(inputs=conv1,epsilon=1e-4)
	maxpool1 = tf.layers.max_pooling2d(conv1, pool_size=(2,2), strides=(2,2), padding='same')
	# Now 150x200x16

	conv2 = tf.layers.conv2d(inputs=maxpool1, filters=8, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
	# Now 150x200x8
	conv2 =tf.layers.batch_normalization(inputs=conv2,epsilon=1e-4)
	maxpool2 = tf.layers.max_pooling2d(conv2, pool_size=(2,2), strides=(2,2), padding='same')
	# Now 75x100x8

	conv3 = tf.layers.conv2d(inputs=maxpool2, filters=8, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
	# Now 75x100x8
	conv3 =tf.layers.batch_normalization(inputs=conv3,epsilon=1e-4)
	maxpool3 = tf.layers.max_pooling2d(conv3, pool_size=(2,2), strides=(2,2), padding='same')
	# Now 38x50x8

	conv4 = tf.layers.conv2d(inputs=maxpool3, filters=8, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
	encoded = tf.layers.max_pooling2d(conv4, pool_size=(2,2), strides=(2,2), padding='same')
	#Now 19x25x8

	return encoded

### Decoder
def decoder(encoded):
	upsample1 = tf.image.resize_images(encoded, size=(38,50), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
	# Now 38x50x8
	conv5 = tf.layers.conv2d(inputs=upsample1, filters=8, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
	# Now 38x50x8
	conv5 =tf.layers.batch_normalization(inputs=conv5,epsilon=1e-4)


	upsample2 = tf.image.resize_images(conv5, size=(75,100), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
	# Now 75x100x8
	conv6 = tf.layers.conv2d(inputs=upsample2, filters=8, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
	# Now 75x100x8
	conv6 =tf.layers.batch_normalization(inputs=conv6,epsilon=1e-4)


	upsample3 = tf.image.resize_images(conv6, size=(150,200), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
	# Now 150x200x8
	conv7 = tf.layers.conv2d(inputs=upsample3, filters=8, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
	# Now 150x200x8
	conv7 =tf.layers.batch_normalization(inputs=conv7,epsilon=1e-4)


	upsample4 = tf.image.resize_images(conv7, size=(300,400), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
	# Now 300x400x8
	conv8 = tf.layers.conv2d(inputs=upsample4, filters=16, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
	# Now 300x400x16
	conv8 =tf.layers.batch_normalization(inputs=conv8,epsilon=1e-4)

	
	decoded = tf.layers.conv2d(inputs=conv8, filters=1,	 kernel_size=(3,3), padding='same', activation=tf.nn.relu)
	#Now 400x300x3
	#decoded = tf.nn.sigmoid(logits)

	return decoded

#autoencoder
def model(x):
	encoded=encoder(x)
	decoded=decoder(encoded)
	return decoded

#function for training
def train(args):
	data_path=args.train_data_path

	x_train=np.load(data_path+"x_train.npy")
	y_train=np.load(data_path+"y_train.npy")

	n_train=x_train.shape[0]

	#hyperparameter
	learning_rate = 0.001
	# Input and target placeholders
	X= tf.placeholder(tf.float32, (None, 300,400,3), name="input")
	Y= tf.placeholder(tf.float32, (None, 300,400,1), name="target")


	#loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=decoded,labels=Y)
	decoded=model(X)
	cost= tf.reduce_mean(tf.square(decoded-Y)) * 0.5

	#cost = tf.reduce_mean(loss)  #cost
	opt = tf.train.AdamOptimizer(learning_rate).minimize(cost)



	#saver_path="/home/dilip/Documents/course/6thsem/DL/Assignment/Assignment3/q2/tmp/"4
	saver_path=args.model_path
	with tf.Session() as sess:
		epochs = 250
		batch_size =15
		path="output/"
		saver = tf.train.Saver()
		sess.run(tf.global_variables_initializer())
		for e in range(epochs):
			i=0;batch_num=0
			while i<n_train:
				x_,y_,i=next_batch(x_train,y_train,i,batch_size)
				#print("batch taken")

				batch_cost= sess.run(cost, feed_dict={X: x_,Y: y_})
				_= sess.run(opt, feed_dict={X: x_,Y: y_})	
				#print("loss calculated")
				batch_num+=1
				print("epoch=",e,"batch_num=",batch_num,"loss=",batch_cost)
			if e==epochs-1:
				save_path = saver.save(sess, saver_path+"model"+str(e+1))
				print("Model saved in path: %s" % save_path)

		x_,y_,_=next_batch(x_train,y_train,0,batch_size)
		x1=y_[0]
		dec=sess.run(decoded,feed_dict={X: x_,Y: y_})
		x2=np.array(dec[0])

		cv2.imwrite(path+"actual"+str(e+1)+".png",x1)
		cv2.imwrite(path+"predicted"+str(e+1)+".png",x2)

#function for testing
def test(args):
	data_path=args.test_data_input_path
	x_test,y_path=load_test_data(data_path)

	predicted_mask=args.mask_output_path
	n_test=x_test.shape[0]


	#hyperparameters
	learning_rate = 0.001
	batch_size =10
	# Input and target placeholders
	X= tf.placeholder(tf.float32, (None, 300,400,3), name="input")
	# #Y= tf.placeholder(tf.float32, (None, 2), name="target")
	out=model(X)
	#tf.reset_default_graph()

	# import the graph from the file
	#saver_path="/home/dilip/Documents/course/6thsem/DL/Assignment/Assignment3/q2/tmp/"
	saver_path=args.model_path
	#imported_graph = tf.train.import_meta_graph(saver_path+"model1.meta")

	saver = tf.train.Saver()

	#saver_path="/home/dilip/Documents/course/6thsem/DL/Assignment/Assignment3/q3/tmp/"
	with tf.Session() as sess:
		saver.restore(sess, saver_path+"model250")
		i=0
		print("in session")
		while i<n_test:  
		    x,y_,i=next_test_batch(x_test,y_path,i,batch_size)
		    x_=sess.run(out, feed_dict={X: x})
		    save_predicted_mask(predicted_mask,x_,y_)

	

#main function of the program
def main(args):
	if args.phase=="train":
		train(args)
	else:
		test(args)

#parsing argument and calling main function
args = parser.parse_args()
main(args)	
