#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os

dirname = 'C://Users//DELL NOTEBOOK//Desktop//AI'
files = os.listdir(dirname)

print(files)


# In[2]:


train=[]
test=[]
for file in files:
    path=dirname+'//{}'.format(file)
    print(path)
    images=os.listdir(path)
    print(images)
    no_of_images=len(images)
    print(no_of_images)
    test_train_ratio=0.10
    no_of_images_in_train=int((1-test_train_ratio)*no_of_images)
    print(no_of_images_in_train)
    for i in range(no_of_images_in_train):
        row=[]
        print(path+"//{}.jpg".format(images[i]))
        row.append(path+"//{}".format(images[i]))
        #label,temp=images[i].split("_")
        #print(label))
        row.append(file)
        train.append(row)
    for i in range(no_of_images_in_train,no_of_images):
        row=[]
        row.append(path+"//{}".format(images[i]))
        #label,temp=images[i].split("_")
        #print(label))
        row.append(file)
        test.append(row)
    


# In[3]:


import random
random.shuffle(train)
print(train)


# In[4]:


import cv2
import numpy as np
X_train=[]
Y_train=[]
for row in train:
    path=row[0]
    Y_train.append(row[1])
    img=cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    print(type(img))
    print(img.shape)
    X_train.append(img)
print(len(X_train))
X_train=np.array(X_train)
#Y_train=np.array(Y_train)


# In[5]:


import cv2
import numpy as np
X_test=[]
Y_test=[]
for row in test:
    print(row)
    path=row[0]
    print(path)
    Y_test.append(row[1])
    img=cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    #print(img)
    print(type(img))
    print(img.shape)
    X_test.append(img)
print(len(X_test))
print(len(Y_test))
X_test=np.array(X_test)
#Y_test=np.array(Y_test)


# In[6]:


print(Y_test)


# In[7]:


import keras 
from keras.datasets import mnist 
from keras.models import Sequential 
from keras.layers import Dense, Dropout, Flatten 
from keras.layers import Conv2D, MaxPooling2D 
from keras import backend as K 
import numpy as np


# In[8]:


x_train=X_train
y_train=Y_train
x_test=X_test
y_test=Y_test


# In[9]:


#from keras.utils import to_categorical
from tensorflow.keras.utils import to_categorical
img_rows, img_cols = 160,160

if K.image_data_format() == 'channels_first': 
   x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols) 
   x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols) 
   input_shape = (1, img_rows, img_cols) 
else: 
   x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1) 
   x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1) 
   input_shape = (img_rows, img_cols, 1) 
   
x_train = x_train.astype('float32') 
x_test = x_test.astype('float32') 
x_train /= 255 
x_test /= 255 

y_train = to_categorical(y_train, 15) 
y_test = to_categorical(y_test, 15)


# In[10]:


model = Sequential() 
model.add(Conv2D(32, kernel_size = (3, 3),  
   activation = 'relu', input_shape = input_shape)) 
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Conv2D(64, (3, 3), activation = 'relu')) 
model.add(MaxPooling2D(pool_size = (2, 2))) 
model.add(Dropout(0.25))
model.add(Flatten()) 
model.add(Dense(128, activation = 'relu')) 
model.add(Dropout(0.5)) 
model.add(Dense(15, activation = 'softmax'))


# In[11]:


model.compile(loss = keras.losses.categorical_crossentropy, 
   optimizer = 'adam', metrics = ['accuracy'])


# In[13]:


training = model.fit(
   x_train, y_train, 
   batch_size = 128, 
   epochs = 12, 
   verbose = 1, 
   validation_data = (x_test, y_test)
)


# In[14]:


score = model.evaluate(x_test, y_test, verbose = 0)

print('Test loss:', score[0]) 
print('Test accuracy:', score[1])


# In[15]:


score = model.evaluate(x_test, y_test, verbose = 0)

print('Test loss:', score[0]) 
print('Test accuracy:', score[1])


# In[16]:


import matplotlib.pyplot as plt

# Extract the history from the training object
history = training.history

# Plot the training loss 
plt.plot(history['loss'])
# Plot the validation loss
plt.plot(history['val_loss'])

# Show the figure
plt.show()

