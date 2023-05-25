#!/usr/bin/env python
# coding: utf-8

# In[78]:


#Loading the libraries

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras import Sequential
from keras.models import load_model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter)
from dataclasses import dataclass
import time
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from keras.optimizers import Adam


# In[79]:


#Loading the data
data = pd.read_csv("C:/Users/14692/Desktop/project/Untitled Folder/archive/data.csv")


# In[80]:


data.head(n=6)
print(len(data))


# In[81]:


#Images count of various characters
data.groupby("character").count()


# In[82]:


#List of unique characters & digits in our dataset

char_names = data.character.unique()  
rows =10;columns=5;
fig, ax = plt.subplots(rows,columns, figsize=(8,16))
for row in range(rows):
    for col in range(columns):
        ax[row,col].set_axis_off()
        if columns*row+col < len(char_names):
            x = data[data.character==char_names[columns*row+col]].iloc[0,:-1].values.reshape(32,32)
            x = x.astype("float64")
            x/=255
            ax[row,col].imshow(x, cmap="binary")
            ax[row,col].set_title(char_names[columns*row+col].split("_")[-1])

            
plt.subplots_adjust(wspace=1, hspace=1)   
plt.show()


# In[83]:


#Distribution of pixel values in the dataset.

import matplotlib.pyplot as plt
plt.hist(data.iloc[0,:-1])
plt.show()


# In[84]:


#Normalizing the Data values\
X = data.values[:,:-1]/255.0
Y = data["character"].values


# In[85]:


print(type(X))
print(Y)


# In[86]:


del data
n_classes = 46


# In[87]:


#Splitting the dataset to train & test
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.15)

# Encode the categories
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)
y_train = to_categorical(y_train, n_classes)
y_test = to_categorical(y_test, n_classes)


# In[88]:


img_height_rows = 32
img_width_cols = 32


# In[89]:


x_train.shape
print(y_train.shape)
type(y_train)
y_train


# In[90]:


im_shape = (img_height_rows, img_width_cols, 1)
x_train = x_train.reshape(x_train.shape[0], *im_shape) # Python TIP :the * operator unpacks the tuple
x_test = x_test.reshape(x_test.shape[0], *im_shape)


# In[91]:


#Splitting Train to Trian & Validation
x_train1, x_val, y_train1, y_val = train_test_split(x_train, y_train, test_size=0.18)


# In[92]:


#Defining the CNN model

from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D 
Name = "model2-cnn-64x2-{}".format(int(time.time()))
tensorboard = TensorBoard(log_dir='logs/{}'.format(Name))

input_shape=(32, 32, 1)
own_model = Sequential()

own_model.add(Conv2D(64, (3,3), input_shape= (32, 32, 1)))
own_model.add(Activation("relu"))
own_model.add(Conv2D(64, (3,3)))
own_model.add(Activation("relu"))
own_model.add(MaxPooling2D(pool_size=(2,2)))

own_model.add(Dropout(0.2))

own_model.add(Conv2D(128, (3,3)))
own_model.add(Activation("relu"))
own_model.add(Conv2D(128, (3,3)))
own_model.add(Activation("relu"))
own_model.add(MaxPooling2D(pool_size=(2,2)))

own_model.add(Dropout(0.2))

own_model.add(Flatten())
own_model.add(Dense(64))
own_model.add(Activation('relu'))
own_model.add(Dense(64))
own_model.add(Activation('relu'))

own_model.add(Dense(46))
own_model.add(Activation('softmax'))


# In[93]:


own_model.summary()


# In[94]:


#Compilng the model

own_model.compile( loss="categorical_crossentropy",
               optimizer="adam",
               metrics=["accuracy"])


# In[95]:


#Converting Data to tensorflow

x_train1 = tf.constant(x_train1, dtype=tf.float32)
y_train1 = tf.constant(y_train1, dtype=tf.float32)

#Fitting the model
history1 = own_model.fit(x_train1, y_train1, batch_size=32, epochs=15,callbacks =[tensorboard])


# In[99]:


#Plotting the Training Accuracy 

plt.plot(history1.history['accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()


# In[98]:


#Plotting the Model Loss


plt.plot(history1.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()


# In[100]:


#Model Accuracy - Loss Graph

plt.plot(history1.history['accuracy'])
plt.plot(history1.history['loss'])
plt.title('model loss-acc')
plt.ylabel('loss-accuracy')
plt.xlabel('epoch')
plt.legend(['accuracy', 'los'], loc='upper left')
plt.show()


# In[101]:


own_model.save('model2.model')


# In[102]:


x_val = tf.constant(x_val, dtype=tf.float32)
y_val = tf.constant(y_val, dtype=tf.float32)


# In[103]:


#Evalauting the model on Valdiation Dataset
score = own_model.evaluate(x_val, y_val, batch_size=32)


# In[104]:


x_test = tf.constant(x_test, dtype=tf.float32)
y_test = tf.constant(y_test, dtype=tf.float32)


# In[105]:


#Evalauting the model on Test Dataset
score = own_model.evaluate(x_test, y_test, batch_size=32)


# In[106]:


#Predicting the test values
ypred = own_model.predict(x_test)


# In[121]:


from pandas.core.generic import FilePath
import os
import cv2
from PIL import Image

# specify the path to the root folder
#filepath = 'C:/Users/14692/Desktop/project/Untitled Folder/archive/Images/Images/character_35_tra/3424.png'

filepath = 'C:/Users/14692/Desktop/project/Untitled Folder/data/wrong/5.png'

Img_size = 32
#Loading & resizing the dataset
img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
new_array = cv2.resize(img_array, (Img_size, Img_size))

print("INPUT CHARACTER")

#Plotting input image
plt.imshow(new_array, cmap='gray')
plt.show()

#Loading the model
x = new_array.reshape(-1, Img_size, Img_size, 1)
model = tf.keras.models.load_model('model2.model')

#Predicting the model
prediction = model.predict(x)
prediction=prediction[0]

#using Armax to get the index value & priting the character or digit name 
max_index = np.argmax(prediction)
catogs = char_names
r = catogs[max_index]
print("The Predicted Character :-",r)

