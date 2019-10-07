

from __future__ import print_function

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

import tensorflow as tf
import numpy as np
import layer

import matplotlib.pyplot as plt 



#train_x=np.load('data_q3/train/train_x.npy')
#print(train_x.shape)
#train_y=np.load('data_q3/train/train_y.npy')
#print(train_y.shape)
#test_x=np.load('')
#test_y=np.load('')


# for mnist dataset, uncomment the following




old_train_x=np.load('mnist/train_x.npy')
old_train_y=np.load('mnist/train_y.npy')
old_test_x=np.load('mnist/test_x.npy')
old_test_y=np.load('mnist/test_y.npy')


#creating input for training

train_x=old_train_x.reshape(old_train_x.shape[0],784)
test_x=old_test_x.reshape(old_test_x.shape[0],784)
print(train_x.shape,test_x.shape)
train_y=np.zeros((old_train_y.shape[0],10))
test_y=np.zeros((old_test_y.shape[0],10))

for i in range(train_y.shape[0]):
    train_y[i][int(old_train_y[i])]=1
        
for i in range(test_y.shape[0]):
	test_y[i][int(old_test_y[i])]=1

'''

train_x=np.load('test/tr_x.npy')
train_y=np.load('test/tr_y.npy')
test_x=np.load('test/te_x.npy')
test_y=np.load('test/te_y.npy')
print(train_x.shape)



'''






# Construct model
model=layer.mylayer(num_classes=96,num_input=784,learning_rate=0.1,hidden_1=256,hidden_2=128)
X = tf.placeholder("float")
Y = tf.placeholder("float")
logits = model.neural_net(X)

y_pred = tf.nn.softmax(logits)
y_pred_cls = tf.argmax(y_pred, axis=1)
y_real=tf.argmax(Y,axis=1)
con=tf.confusion_matrix(labels=y_real, predictions=y_pred_cls, num_classes=model.num_classes)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=model.learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model (with test logits, for dropout to be disabled)
correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

saver=tf.train.Saver()

# Start training
with tf.Session() as sess:
	sess.run(init)
	i=0
	loss_arr=np.zeros(model.num_steps)
	acc_arr=np.zeros(model.num_steps)
	iteration=np.zeros(model.num_steps)

	for step in range(1, model.num_steps+1):
		# batch_x, batch_y = mnist.train.next_batch(model.batch_size)
		# print(train_x.shape)
		#Run optimization op (backprop)
		sess.run(train_op, feed_dict={X: train_x, Y: train_y})
		if step % model.display_step == 0 or step == 1:
			# Calculate batch loss and accuracy
			loss, acc = sess.run([loss_op, accuracy], feed_dict={X: train_x,Y: train_y})
			print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))
			loss_arr[i]=loss
			acc_arr[i]=acc
			iteration[i]=i+1
			i+=1

	print("Optimization Finished!")
	save_path = saver.save(sess, "my_model")
	i=i-1
	print("Testing Accuracy:", sess.run(accuracy, feed_dict={X: test_x,Y: test_y}))
	print("confusion matrix:",sess.run(con,feed_dict={X: train_x,Y: train_y}))
	#print("F_score :",sess.run(score,feed_dict={X: train_x,Y: train_y}))
	#print(correct_pred)
	#con=tf.confusion_matrix(train_y,correct_pred)
	#tf.io.write_file(con_mnist.txt,con)
	plt.plot(iteration[:i],loss_arr[:i])
	plt.ylabel("loss")
	plt.xlabel("no of iterations")
	plt.show()
	plt.plot(iteration[:i],acc_arr[:i])
	plt.xlabel("no of iterations")
	plt.ylabel("accuracy")
	plt.show()
	print(con)
	#print(score)
	

