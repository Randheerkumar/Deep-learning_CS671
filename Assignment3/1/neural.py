
'''

@author : Randheer kumar
Deep learning : cs671

'''

import tensorflow as tf 
import numpy as np 







def feature_ext(inputs):
	#input2d = tf.reshape(inputs, [-1,image_size,image_size,1])

	conv1 = tf.layers.conv2d(inputs=inputs, filters=10, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)
	pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

	#return pool1
	
	conv2 = tf.layers.conv2d(inputs=pool1, filters=10, kernel_size=[2, 2], padding="same", activation=tf.nn.relu)
	#pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

	return conv2 
	

'''
#fully connected layer1
def dense_net1(inputs,num_classes):
	dim=1
	for i in range(1,len(inputs.shape)):
		dim*=inputs.shape[i]
		#print(inputs.shape[i])
	#print(dim)
	
	pool_flat= tf.reshape(inputs, [-1, dim])	
	hidden = tf.layers.dense(inputs= pool_flat, units=56, activation=tf.nn.relu)
	#hidden= tf.layers.dropout(hidden, rate=dropout, training=is_training)
	output = tf.layers.dense(inputs=hidden, units=num_classes)
    #output=tf.nn.sigmoid(output, name ='sigmoid') 

	return output

'''
def dense_net2(inputs,num_of_output):
	dim=1
	for i in range(1,len(inputs.shape)):
		dim*=inputs.shape[i]
		
	pool_flat= tf.reshape(inputs, [-1, dim])	
	hidden = tf.layers.dense(inputs= pool_flat, units=56, activation=tf.nn.relu)
	#hidden= tf.layers.dropout(hidden, rate=dropout, training=is_training)
	output = tf.layers.dense(inputs=hidden, units=num_of_output)
    #output=tf.nn.sigmoid(output, name ='sigmoid') 

	return output






def cnn(inputs,num_classes):
	conv_out=feature_ext(inputs)
	out=dense_net1(conv_out,num_classes)
	return out

def regression_head(inputs,num_of_output):
	output=dense_net2(inputs,num_of_output)
	return output




def create_batch(i,x,y,gy,batch_size):
	
	batch_x=x[i:min(i+batch_size,x.shape[0])]
	batch_y=y[i:min(i+batch_size,y.shape[0])]
	batch_gy=gy[i:min(i+batch_size,y.shape[0])]
	return batch_x,batch_y,batch_gy


def task2_batch(i,x,y,batch_size):
	
	batch_x=x[i:min(i+batch_size,x.shape[0])]
	batch_y=y[i:min(i+batch_size,y.shape[0])]
	return batch_x,batch_y
