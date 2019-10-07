
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
import IntersectionOverUnion as iou 

import model



#hyperparameter
learning_rate=0.01
batch_size=32
epochs=30


num_classes=3  #number of classes
num_box=1   #number of object in a image



#function for one hot encoding
def one_hot(x):
	arr=np.zeros((x.shape[0],num_classes))
	for i in range(x.shape[0]):
		arr[i][int(x[i])]=1
	return arr




x=tf.placeholder(tf.float32,shape=(None,288,352,1))
y=tf.placeholder(tf.float32,shape=(None,3))
ground_truth=tf.placeholder(tf.float32,shape=(None,num_box*4))


#training dataset
train_x=np.load("x_train.npy")   	#image dataset
train_y=np.load("y_train.npy")  	#class label
train_gy=np.load("gy_train.npy")    #ground truth

#testing dataset
test_x=np.load("x_test.npy")
test_y=np.load("y_test.npy")
test_gy=np.load("gy_test.npy")            

#one hot encoding
train_y=one_hot(train_y)
test_y=one_hot(test_y)





#classification
score1=model.cnn(x,num_classes)

#localization, regression head
predicted_cord=model.regression_head(model.feature_ext(x),num_box*4)

#cross entropy for classification
cross_entrop_loss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=score1,labels=y))

#regression loss for localization
regression_loss=tf.reduce_mean(tf.squared_difference(ground_truth, predicted_cord))


#prediction of class
prediction=tf.nn.softmax(score1)


correct_prediction=tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
accuracy1=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

#loss optimizer
optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op=optimizer.minimize(cross_entrop_loss+regression_loss)






#initialzing variable in tensorflow
init=tf.global_variables_initializer()

result=np.zeros((test_x.shape[0],4))

jj=0

with tf.Session() as sess:
	sess.run(init);
	print("training starts\n")
	for step in range(epochs):
		i=0
		entropy_loss=0
		reg_loss=0
		accuracy=0
		count=0
		while i < train_x.shape[0]:
			batch_x,batch_y,batch_gy=model.create_batch(i,train_x,train_y,train_gy,batch_size)
			i+=batch_size
			sess.run(train_op,feed_dict={x:batch_x,y:batch_y,ground_truth:batch_gy})
			batch_entropy_loss,batch_accuracy,reg_batch_loss=sess.run([cross_entrop_loss,accuracy1,regression_loss], feed_dict={x:batch_x,y:batch_y,ground_truth:batch_gy})
			
			entropy_loss+=batch_entropy_loss
			reg_loss+=reg_batch_loss
			accuracy+=batch_accuracy
			count+=1
			#print("regression loss ",reg_batch_loss)
			#print("accuracy is :", batch_accuracy)
			#print("cross entropy loss:",batch_entropy_loss)


		
		print("Step " + str(step) + ", epoch entropy Loss= " + \
			"{:.4f}".format(entropy_loss/count) + ", Training Accuracy= " + \
			"{:.3f}".format(accuracy/count) +"regression loss=" + "{:.1f}".format(reg_loss/count))



	print("training finished\n")
	print("testing starts\n")
	i=0
	j=0
	test_reg_loss=0
	test_accuracy=0
	test_entropy_loss=0
	count=0
	jj=0
	while i < test_x.shape[0]:
		#print("i is ",i)
		batch_x,batch_y,batch_gy=model.create_batch(i,test_x,test_y,test_gy,batch_size)
		i+=batch_size
		batch_entropy_loss,batch_accuracy,reg_batch_loss=sess.run([cross_entrop_loss,accuracy1,regression_loss], feed_dict={x:batch_x,y:batch_y,ground_truth:batch_gy})
		print(jj)
		result[jj:min(jj+batch_size,test_x.shape[0]),:]=sess.run(predicted_cord,feed_dict={x:batch_x,y:batch_y,ground_truth:batch_gy})
		
		
		jj+=batch_size
		test_entropy_loss+=batch_entropy_loss
		test_reg_loss+=reg_batch_loss
		test_accuracy+=batch_accuracy
		count+=1
		print("batch regression loss ",reg_batch_loss)
		print("batch accuracy is :", batch_accuracy)
		print("batch cross entropy loss:",batch_entropy_loss)


	
	print(", testing entropy Loss= " + \
		"{:.4f}".format(entropy_loss/count) + ", Testing Accuracy= " + \
		"{:.3f}".format(test_accuracy/count) +"test regression loss=" + "{:.1f}".format(test_reg_loss/count))

ans=0
temp=0
file=open("task1.txt","w")
for i in range(test_x.shape[0]):
	boxA=result[i]
	boxB=test_gy[i]
	temp=iou.bb_intersection_over_union(boxA, boxB)
	print(temp)
	file.write(str(temp))
	if temp > 0.5 :
		ans+=1
print("iou greater than 0.5",ans)
print("total number of training data",test_x.shape[0])
file.close()







