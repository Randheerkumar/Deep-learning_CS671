
#importing various libraries
import numpy as np
import tensorflow as tf


#loading the dataset
x_train=np.load('x1_train.npy')
x_test=np.load('x1_test.npy')
y_train=np.load('y1_train.npy')
y_test=np.load('y1_test.npy')



#this function return the data of next batch
def next_batch(i,x,y1,y2,y3,y4,batch_size):
    n=x.shape[0]
    if(i+batch_size<n):
        x_=x[i:i+batch_size,:,:,:]
        y_1=y1[i:i+batch_size]
        y_2=y2[i:i+batch_size]
        y_3=y3[i:i+batch_size]
        y_4=y4[i:i+batch_size,:]


    else:
        x_=x[i:,:,:,:]
        y_1=y1[i:]
        y_2=y2[i:]
        y_3=y3[i:]
        y_4=y4[i:,:]

    return x_,y_1,y_2,y_3,y_4,i+batch_size   
 

#this function is used for one hot encoding
def one_hot(y,c):
    n=y.shape[0]
    Y=np.zeros((n,c))
    for i in range(n):
        Y[i][int(y[i])]=1

    return(Y)        

 
#function calculates the confusion magtrix 
def confusion_matrix(conf,pred,label):
    n=pred.shape[0]
    for i in range(n):
        conf[int(label[i])][int(pred[i])]+=1 

    return conf 

def confusion_matrix_onehot(conf,pred,label):
    n=pred.shape[0]
    for i in range(n):
        conf[np.argmax(label[i])][np.argmax(pred[i])]+=1 

    return conf     


#hyperparameters
num_classes1=1
num_classes2=1
num_classes3=1
num_classes4=12


image_size=28
num_steps=10
batch_size=128
display_step=10
learning_rate=0.01
n=x_train.shape[0]
n1=x_test.shape[0]


#here is label in one hot encoding form
#y1_train=one_hot(y_train[:,0],2)
y1_train=y_train[:,0]
y2_train=y_train[:,1]
y3_train=y_train[:,3]
y4_train=one_hot(y_train[:,2],12)

y1_test=y_test[:,0]
y2_test=y_test[:,1]
y3_test=y_test[:,3]
y4_test=one_hot(y_test[:,2],12)

#print(y1_test.shape[0])

#these are the placeholder of input feature and labels for each head
X = tf.placeholder(tf.float32, [None, 28,28,3])
Y1 = tf.placeholder(tf.float32, [None, num_classes1])
Y2 = tf.placeholder(tf.float32, [None, num_classes2])
Y3 = tf.placeholder(tf.float32, [None, num_classes3])
Y4 = tf.placeholder(tf.float32, [None, num_classes4])




#convolutional layer
def conv_net(inputs,image_size):
	#input2d = tf.reshape(inputs, [-1,image_size,image_size,1])

	conv1 = tf.layers.conv2d(inputs=inputs, filters=32, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)
	pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

	conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
	pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

	return pool2



#fully connected layer1
def dense_net1(inputs,num_classes,is_training):
	pool_flat= tf.reshape(inputs, [-1, 7 * 7 * 64])	
	hidden = tf.layers.dense(inputs= pool_flat, units=1024, activation=tf.nn.relu)
	#hidden= tf.layers.dropout(hidden, rate=dropout, training=is_training)
	output = tf.layers.dense(inputs=hidden, units=num_classes)
    #output=tf.nn.sigmoid(output, name ='sigmoid') 

	return output

#fully connected layer2
def dense_net2(inputs,num_classes,is_training):
    pool_flat= tf.reshape(inputs, [-1, 7 * 7 * 64]) 
    hidden = tf.layers.dense(inputs= pool_flat, units=1024, activation=tf.nn.relu)
    #hidden= tf.layers.dropout(hidden, rate=dropout, training=is_training)
    output = tf.layers.dense(inputs=hidden, units=num_classes)

    return output

#fully connected layer3
def dense_net3(inputs,num_classes,is_training):
    pool_flat= tf.reshape(inputs, [-1, 7 * 7 * 64]) 
    hidden = tf.layers.dense(inputs= pool_flat, units=1024, activation=tf.nn.relu)
    #hidden= tf.layers.dropout(hidden, rate=dropout, training=is_training)
    output = tf.layers.dense(inputs=hidden, units=num_classes)

    return output

#fully connected layer4
def dense_net4(inputs,num_classes,is_training):
    pool_flat= tf.reshape(inputs, [-1, 7 * 7 * 64]) 
    hidden = tf.layers.dense(inputs= pool_flat, units=1024, activation=tf.nn.relu)
    #hidden= tf.layers.dropout(hidden, rate=dropout, training=is_training)
    output = tf.layers.dense(inputs=hidden, units=num_classes)

    return output



#our model
def model1(inputs,num_classes,image_size,is_training):
	conv_out=conv_net(inputs,image_size)
	out=dense_net1(conv_out,num_classes,is_training)
	return out

def model2(inputs,num_classes,image_size,is_training):
    conv_out=conv_net(inputs,image_size)
    out=dense_net2(conv_out,num_classes,is_training)
    return out

def model3(inputs,num_classes,image_size,is_training):
    conv_out=conv_net(inputs,image_size)
    out=dense_net3(conv_out,num_classes,is_training)
    return out

def model4(inputs,num_classes,image_size,is_training):
    conv_out=conv_net(inputs,image_size)
    out=dense_net4(conv_out,num_classes,is_training)
    return out            




#classification head1
output1=model1(X,num_classes1,image_size,True) 

loss1= tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output1, labels=Y1))

prediction1 = tf.round(tf.nn.sigmoid(output1))
correct_prediction1 = tf.equal(prediction1,Y1)
accuracy1 = tf.reduce_mean(tf.cast(correct_prediction1, tf.float32))
#conf1=tf.confusion_matrix(tf.reshape(Y1,[Y1.shape[0]]),tf.reshape(prediction1,[prediction1.shape[0]]),num_classes=2)

#classification head2
output2=model2(X,num_classes2,image_size,True) 

loss2= tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output2, labels=Y2))

prediction2 = tf.round(tf.nn.sigmoid(output2))
correct_prediction2 = tf.equal(prediction2,Y2)
accuracy2 = tf.reduce_mean(tf.cast(correct_prediction2, tf.float32))

#classification head3
output3=model3(X,num_classes3,image_size,True) 

loss3= tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output3, labels=Y3))

prediction3 = tf.round(tf.nn.sigmoid(output3))
correct_prediction3 = tf.equal(prediction3,Y3)
accuracy3 = tf.reduce_mean(tf.cast(correct_prediction3, tf.float32))


#classification head4
output4=model4(X,num_classes4,image_size,True) 
loss4= tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output4, labels=Y4))

prediction4 = tf.nn.softmax(output4)
correct_prediction4 = tf.equal(tf.argmax(prediction4, 1), tf.argmax(Y4, 1))
accuracy4 = tf.reduce_mean(tf.cast(correct_prediction4, tf.float32))




loss=0.3*loss1+0.3*loss2+0.25*loss3+0.15*loss4
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op= optimizer.minimize(loss1)



#here the session begins
init = tf.global_variables_initializer()

saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(init);
    curve=np.zeros((num_steps,6))
    for step in range(1, num_steps+1):
        i=0;ac1=0;ac2=0;ac3=0;ac4=0;los1=0;count=0
        while i<n:
            batch_x,y_1,y_2,y_3,y_4,i=next_batch(i,x_train,y1_train,y2_train,y3_train,y4_train,batch_size)
            y_11=np.reshape(y_1,[128,1])
            y_22=np.reshape(y_2,[128,1])
            y_33=np.reshape(y_3,[128,1])

            sess.run(train_op, feed_dict={X: batch_x, Y1: y_11, Y2: y_22, Y3: y_33, Y4: y_4})
                #if step % display_step == 0 or step == 1:
            los, acc1,acc2,acc3,acc4= sess.run([loss, accuracy1,accuracy2,accuracy3,accuracy4], feed_dict={X: batch_x,Y1: y_11, Y2: y_22, Y3: y_33, Y4: y_4})
            ac1+=acc1
            ac2+=acc2
            ac3+=acc3
            ac4+=acc4
            los1+=los
            count+=1

        curve[step-1][0]=step
        curve[step-1][1]=los1/count
        curve[step-1][2]=ac1/count
        curve[step-1][3]=ac2/count
        curve[step-1][4]=ac3/count
        curve[step-1][5]=ac4/count
        print("Step " + str(step) + ", epoch Loss= " + \
              "{:.4f}".format(los1/count) + ", Training Accuracy= " + \
              "{:.3f}".format(ac1/count)+"{:.3f}".format(ac2/count)+"{:.3f}".format(ac3/count)+"{:.3f}".format(ac4/count))

    save_path = saver.save(sess, "tmp/model.ckpt")
    print("Model saved in path: %s" % save_path)

    print("Optimization Finished!")

    print("testing starts")
    i=0;ac1=0;ac2=0;ac3=0;ac4=0;count=0
    conf1=np.zeros((2,2))
    conf2=np.zeros((2,2))
    conf3=np.zeros((2,2))
    conf4=np.zeros((12,12))

    while i<n1:
        batch_x,y_1,y_2,y_3,y_4,i=next_batch(i,x_test,y1_test,y2_test,y3_test,y4_test,batch_size)
        y_11=np.reshape(y_1,[128,1])
        y_22=np.reshape(y_2,[128,1])
        y_33=np.reshape(y_3,[128,1])

        pred1,pred2,pred3,pred4=sess.run([prediction1 ,prediction2,prediction3,prediction4], feed_dict={X:batch_x,Y1: y_11, Y2: y_22, Y3: y_33, Y4: y_4})
        acc1,acc2,acc3,acc4= sess.run([accuracy1,accuracy2,accuracy3,accuracy4], feed_dict={X:batch_x,Y1: y_11, Y2: y_22, Y3: y_33, Y4: y_4})
        conf1=confusion_matrix(conf1,pred1,y_1)
        conf2=confusion_matrix(conf2,pred2,y_2)
        conf3=confusion_matrix(conf3,pred3,y_3)
        conf4=confusion_matrix_onehot(conf4,pred3,y_4)

        ac1+=acc1
        ac2+=acc2
        ac3+=acc3
        ac4+=acc4
        count+=1


#savin the conf adn curve
print("ac1=",ac1/count," ac2=",ac2/count,"ac3=",ac3/count," ac4=",ac4/count) 
print("conf=",conf1) 
print("conf=",conf2)   
print("conf=",conf3)   
print("conf=",conf4)

np.save("data/conf1",conf1)
np.save("data/conf2",conf2)  
np.save("data/conf3",conf3)
np.save("data/conf4",conf4)  
np.save("data/curve",curve)   

     

