
'''

cs671 : April 2019
Object detection

@author : Randheer kumar



'''

from __future__ import print_function

import numpy as np 
import tensorflow as tf 
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split

import neural   #importing own function
import IntersectionOverUnion as iou 

#hyper parameter
learning_rate=0.05
batch_size=12
epochs=30

#number of object in a image
num_box=4


#training and testing dataset
train_x=np.load("four_obj/x_train.npy")   #image dataset
train_y=np.load("four_obj/y_train.npy")  #ground truth

test_x=np.load("four_obj/x_test.npy")
test_y=np.load("four_obj/y_test.npy")



x=tf.placeholder(tf.float32,shape=(None,test_x.shape[1],test_x.shape[2],1))
y=tf.placeholder(tf.float32,shape=(None,num_box*4))






print(test_x.shape)


predicted_cord=neural.regression_head(neural.feature_ext(x),num_box*4)

#mean square loss
regression_loss=tf.reduce_mean(tf.squared_difference(y, predicted_cord))

optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate)

train_op=optimizer.minimize(regression_loss)  #optimizer, back propagating





#intializing variables
init=tf.global_variables_initializer()


result=np.zeros((test_x.shape[0],16))
with tf.Session() as sess:
	sess.run(init);

	print("taining starts")
	step=0
	for step in range(epochs):
		i=0
		loss=0
		count=0
		batch_reg_loss=0
		while i < train_x.shape[0]:
			batch_x,batch_y=neural.task2_batch(i,train_x,train_y,batch_size)
			i+=batch_size

			sess.run(train_op,feed_dict={x:batch_x,y:batch_y}) #updating weights for minimizing loss

			batch_reg_loss=sess.run(regression_loss, feed_dict={x:batch_x,y:batch_y})
			loss+=batch_reg_loss
			count+=1
			print("batch loss ",batch_reg_loss)
			

		
		print("regression loss=" + "{:.1f}".format(loss/count))

	print("testing starts")
	i=0
	test_loss=0
	bacth_loss=0
	count=0
	j=0
	while i < test_x.shape[0]:
		batch_x,batch_y=neural.task2_batch(i,test_x,test_y,batch_size)
		i += batch_size
		result[j:min(j+batch_size,test_x.shape[0]),:]=sess.run(predicted_cord,feed_dict={x:batch_x,y:batch_y})
		j+=batch_size
		bacth_loss = sess.run(regression_loss,feed_dict={x:batch_x, y:batch_y})
		test_loss +=bacth_loss
		count += 1
		print("testing batch loss",bacth_loss)

	print("testing regression loss=" + "{:.1f}".format(test_loss/count))


i=0
for i in range(test_x.shape[0]):
	for j in range(4):
		boxA=result[i,j*4:(j*4)+4]
		boxB=test_y[i,j*4:(j*4)+4]
		temp=iou.bb_intersection_over_union(boxA,boxB)
		print(temp)



