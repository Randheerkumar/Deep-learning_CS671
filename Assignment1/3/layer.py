

import tensorflow as tf
import numpy as np 







#this class creates input layer , hidden layer, and output layer

class mylayer:
    def __init__(self,num_classes,num_input,learning_rate=None,num_steps=None,batch_size=None,display_step=None,hidden_1=None,hidden_2=None):
        
        if learning_rate == None:
            self.learning_rate=0.1
        else :
            self.learning_rate=learning_rate
            
        if num_steps == None :
            self.num_steps=300
        else :
            self.num_steps=num_steps
            
        if batch_size == None :
            self.batch_size=32
        else :
            self.batch_size=batch_size
        
        if display_step == None :
            self.display_step=2
        else :
            self.display_step=display_step
            
        if hidden_1 == None:
            self.hidden_1=100
        else :
            self.hidden_1=hidden_1
        
        if hidden_2 == None :
            self.hidden_2=100
        else :
            self.hidden_2=hidden_2

        self.num_classes=num_classes
        self.num_input=num_input

     
     #this function return output layer   
    def neural_net(self,x):
        learning_rate=self.learning_rate
        num_steps=self.num_steps
        batch_size=self.batch_size
        display_step=self.display_step
        hidden_1=self.hidden_1
        hidden_2=self.hidden_2
        num_input=self.num_input
        num_classes=self.num_classes
        # Storing layers weight and bias
        weights = {
        'h1': tf.Variable(tf.random_normal([num_input, hidden_1])),
        'h2': tf.Variable(tf.random_normal([hidden_1, hidden_2])),
        'out': tf.Variable(tf.random_normal([hidden_2, num_classes]))
        }
        biases = {
        'b1': tf.Variable(tf.random_normal([hidden_1])),
        'b2': tf.Variable(tf.random_normal([hidden_2])),
        'out': tf.Variable(tf.random_normal([num_classes]))
        }
        # Hidden fully connected layer 
        layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
        # Hidden fully connected layer 
        layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
        # Output fully connected layer with a neuron for each class
        out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
        return out_layer

        
        
    