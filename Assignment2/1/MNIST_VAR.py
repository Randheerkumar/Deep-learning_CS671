# imports for array-handling and plotting
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
import math
from keras.utils import np_utils
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report 
import scikitplot as skplt
from sklearn.model_selection import train_test_split


# importing mnist dataset
from keras.datasets import mnist

#load the mnist datasets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

'''
fig = plt.figure()
for i in range(9):
  plt.subplot(3,3,i+1)
  plt.tight_layout()
  plt.imshow(X_train[i], cmap='gray')
  plt.title("Digit: {}".format(y_train[i]))
  plt.xticks([])
  plt.yticks([])
fig
plt.show()
'''

print("X_train shape", X_train.shape)
print("y_train shape", y_train.shape)
print("X_test shape", X_test.shape)
print("y_test shape", y_test.shape)


X_train = X_train.reshape(60000, 28,28,1)
X_test = X_test.reshape(10000, 28,28,1)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')




# normalizing the data to help with the training
train=X_train/255
test = X_test/255

# print the final input shape ready for training
print("Training data shape", train.shape)
print("Testing data shape", test.shape)


# one-hot encoding 
n_classes = 10
print("Shape before one-hot encoding: ", y_train.shape)
Y_train = np_utils.to_categorical(y_train, n_classes)
Y_test = np_utils.to_categorical(y_test, n_classes)
print("Shape after one-hot encoding: ", Y_train.shape)




# Building the network


# Setting up the layers
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (5,5), padding='same', activation=tf.nn.relu, input_shape=(28, 28,1)),  # 7x7 convolutional layer with 32 filters
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(32,(5,5),activation=tf.nn.relu),
    tf.keras.layers.BatchNormalization(),                           # BatchNormalization layer
    tf.keras.layers.Conv2D(32,(5,5), strides=2,activation=tf.nn.relu),    # Strided Convolution with a stride of 2
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Flatten(),                                      # Flattening the input image in a 784 dim. vector
    tf.keras.layers.Dense(1024, activation=tf.nn.relu),             # Fully connected layer with 1024 input units
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10,  activation=tf.nn.softmax)            # Output layer with 10 classes and softmax activation
   
               
])

#compile the model
model.summary()
model.compile(optimizer=tf.train.AdamOptimizer(learning_rate = 0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# training the model and saving metrics in history
history=model.fit(train, Y_train, validation_data=(test,Y_test),epochs=10 ,verbose=1, batch_size=400)
#evaluate accuracy
test_loss,test_accuracy=model.evaluate(test , Y_test)
print("test accuracy is:" , test_accuracy)

## plotting the metrics


#summarize history for accuracy
fig = plt.figure()
plt.subplot(2,1,1)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')


#summarize history for loss
plt.subplot(2,1,2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')

plt.tight_layout()

fig
plt.show()



# We can make predictions now
# load the model and create predictions on the test set

predicted_classes = model.predict_classes(test)

# see which we predicted correctly and which not
correct_indices = np.nonzero(predicted_classes == y_test)[0]
incorrect_indices = np.nonzero(predicted_classes != y_test)[0]
print()
print(len(correct_indices)," classified correctly")
print(len(incorrect_indices)," classified incorrectly")

print("confusion_mat")
print(confusion_matrix(y_test,predicted_classes))
print('cls_report:')
print (classification_report(y_test,predicted_classes))
skplt.metrics.plot_confusion_matrix(y_test, predicted_classes, normalize=True)
plt.show()
