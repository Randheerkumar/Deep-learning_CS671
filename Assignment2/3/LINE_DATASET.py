

#importing the various libraries
from keras.utils import np_utils	
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from keras.layers import BatchNormalization
from keras.callbacks import ModelCheckpoint
from keras import models
import keras
import cv2
import os
import glob
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report 
#import scikitplot as skplt


seed = 7
np.random.seed(seed)

path = '/home/dilip/Documents/course/6thsem/DL/Assignment/Assignment2/question3/Assignment2/1/Image'


#function to load  the data
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


#loadin data
X_data, y_label = data_loader(path)
X_data=X_data[0:10000,:,:,:]
y_label=y_label[0:10000]
print("hiii=",X_data.shape)
print("hii2",y_label.shape)
X_train, X_test, y_train, y_test = train_test_split(X_data, y_label, test_size=0.4, random_state=42)   
X_train = X_train/255
X_test = X_test/255
print("shape=",X_train.shape)
print("Shape before one-hot encoding: ", y_test.shape)
n_classes=96
Y_train = np_utils.to_categorical(y_train, n_classes)
Y_test = np_utils.to_categorical(y_test, n_classes)
print("Shape after one-hot encoding: ", Y_test.shape)



#spliting the dataset into train and validation data


#defining model
model=Sequential()
model.add(Conv2D(32, (5,5), padding='same', activation=tf.nn.relu, input_shape=(28, 28,3)))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2), strides=2))
model.add(Conv2D(16, (3,3), padding='same', activation=tf.nn.relu, input_shape=(14, 14,32)))
model.add(MaxPooling2D((2, 2), strides=2))
model.add(Flatten())
model.add(Dense(1024, activation=tf.nn.relu))
model.add(Dense(96,  activation=tf.nn.softmax))


#model = Convolution_model()
model.summary()
model.compile(optimizer=tf.train.AdamOptimizer(learning_rate = 0.001), loss='categorical_crossentropy', metrics=['accuracy'])
checkpointer = ModelCheckpoint(filepath="best_weights.hdf5", 
                               monitor = 'val_acc',
                               verbose=1, 
                               save_best_only=True)

history = model.fit(X_train, Y_train, epochs=1,callbacks=[checkpointer],validation_data=(X_test,Y_test), batch_size=400, verbose=1)
test_loss,test_accuracy=model.evaluate(X_test, Y_test)
print("test accuracy is:" , test_accuracy)

model.load_weights('best_weights.hdf5')
model.save('shapes_cnn.h5')


#visualising the activation layer

layer_outputs = [layer.output for layer in model.layers[:12]] 
# Extracts the outputs of the top 12 layers
activation_model = models.Model(inputs=model.input, outputs=layer_outputs) # Creates a model that will return these outputs, given the model input
layer_num=1
input_dir="/home/dilip/Documents/course/6thsem/DL/Assignment/Assignment2/question3/Assignment2/1/3/Line/Image_visualisation/"
output_dir="/home/dilip/Documents/course/6thsem/DL/Assignment/Assignment2/question3/Assignment2/1/3/Line/visualise/activation_map/"


def visualise(activations,path,layer):
    layer_activations=activations[layer-1]
    path1=path+"layer"+str(layer)+"/"
    for i in range(layer_activations.shape[3]):
        name=path1+str(i)+".png"
        plt.imsave(name,layer_activations[0,:,:,i])



i=1
for filename in sorted(glob.glob(input_dir+'/*.jpg')):
    img=cv2.imread(filename)
    img=np.expand_dims(img, axis=0)
    activations = activation_model.predict(img)
    print("fvvdvd=",activations[2].shape)
    path1=output_dir+"Image"+str(i)+"/"
    visualise(activations[0],path1,1)
    visualise(activations[2],path1,2)
    i+=1

    UPDATE mysql.user SET authentication_string = PASSWORD('Dilip@1998') WHERE User = 'root'









# img1=cv2.imread("/home/dilip/Documents/course/6thsem/DL/Assignment/Assignment2/question3/Assignment2/1/0_0_0_0_1.jpg")
# img1=img1/255
# img1=np.expand_dims(img1, axis=0)
# activations = activation_model.predict(img1)



# def visualise(activation,path):
	


# layer_activations=activations[layer_num-1]
# out=layer_activations.reshape(112,224)
# plt.imsave("a.png",out)
# for i in range(layer_activations.shape[3]):
# 	name=directory+str(i)+".png"
# 	plt.imsave(name,layer_activations[0,:,:,i])


# first_layer_activation = activations[0]

# for i in range()
# print("layers activation",first_layer_activation.shape)

# plt.imsave('name.png', array)
# plt.imshow(first_layer_activation[0, :, :, 4], cmap='viridis')
# plt.colorbar()
# plt.show()			
