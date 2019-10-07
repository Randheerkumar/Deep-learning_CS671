from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
import keras
import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report 
import scikitplot as skplt
seed = 7
np.random.seed(seed)

path = '/home/himanshi/Workspace/deep-learning/Image'


def data_loader(path):
    train_list = os.listdir(path)

    x_data= []
    y_label = []
    
    for label, element in enumerate(train_list):
        path1 = path + '/' + str(element)
        images = os.listdir(path1)
        for element2 in images:
            path2 = path1 + '/' + str(element2)
            img = cv2.imread(path2)
            x_data.append(img)
            y_label.append(str(label))

        

    x_data = np.asarray(x_data)
    y_label = np.asarray(y_label)
    
    return x_data, y_label

X_data, y_label = data_loader(path)

X_train, X_test, y_train, y_test = train_test_split(X_data, y_label, test_size=0.4, random_state=42)   
X_train = X_train/255
X_test = X_test/255
print("Shape before one-hot encoding: ", y_test.shape)
n_classes=96
Y_train = np_utils.to_categorical(y_train, n_classes)
Y_test = np_utils.to_categorical(y_test, n_classes)
print("Shape after one-hot encoding: ", Y_test.shape)






def Convolution_model():
    model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (7,7), padding='same', activation=tf.nn.relu, input_shape=(28, 28,3)),  # 7x7 convolutional layer with 32 filters
    tf.keras.layers.BatchNormalization(),                           # BatchNormalization layer
    tf.keras.layers.MaxPooling2D((2, 2), strides=2),                # 2x2 maxpooling with a stride of 2
    tf.keras.layers.Flatten(),                                      # Flattening the input image in a 784 dim. vector
    tf.keras.layers.Dense(1024, activation=tf.nn.relu),             # Fully connected layer with 1024 input units
    tf.keras.layers.Dense(96,  activation=tf.nn.softmax)            # Output layer with 10 classes and softmax activation
   
])
    return model


model = Convolution_model()
model.summary()
model.compile(optimizer=tf.train.AdamOptimizer(learning_rate = 0.001), loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, Y_train, epochs=10,validation_data=(X_test,Y_test), batch_size=400, verbose=1)
test_loss,test_accuracy=model.evaluate(X_test, Y_test)
print("test accuracy is:" , test_accuracy)


plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# We can make predictions now
# load the model and create predictions on the test set
y_test= np.array(list(y_test), dtype=int)
predicted_classes = model.predict_classes(X_test)
print("Shape of predicted classes: ", predicted_classes.shape)
print("shape of test classes:", y_test.shape)
# see which we predicted correctly and which not
correct_indices = np.nonzero(predicted_classes == y_test)[0]
incorrect_indices = np.nonzero(predicted_classes != y_test)[0]
print()
print(len(correct_indices)," classified correctly")
print(len(incorrect_indices)," classified incorrectly")
print('predicted_classes=',predicted_classes)
print('true classes=',y_test)
print("confusion_mat")
print(confusion_matrix(y_test,predicted_classes))
print('cls_report:')
print (classification_report(y_test,predicted_classes))
#skplt.metrics.plot_confusion_matrix(y_test, predicted_classes, normalize=True)
#plt.show()
