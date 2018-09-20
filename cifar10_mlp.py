
# coding: utf-8

# In[5]:


import keras
from keras.datasets import cifar10
import numpy as np


# In[2]:


(x_train,y_train),(x_test,y_test) = cifar10.load_data()


# In[3]:


x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255


# In[7]:


num_classes = len(np.unique(y_train))


# In[9]:


y_train = keras.utils.to_categorical(y_train, num_classes)


# In[11]:


y_test = keras.utils.to_categorical(y_test, num_classes)


# In[12]:


(x_train, x_valid) = x_train[5000:], x_train[:5000]
(y_train, y_valid) = y_train[5000:], y_train[:5000]


# In[14]:


from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
model = Sequential()


# In[18]:


model.add(Flatten(input_shape=x_train.shape[1:]))
model.add(Dense(1000, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(500, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))
model.summary()


# In[19]:


model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])


# In[20]:


from keras.callbacks import ModelCheckpoint


# In[21]:


checkpointer = ModelCheckpoint(filepath='MLP.weights.best.hdf5', verbose=1,save_best_only=True)
hist = model.fit(x_train, y_train, batch_size=32, epochs=20, validation_data=(x_valid, y_valid), callbacks=[checkpointer],
                verbose=2, shuffle=True)


# In[ ]:


model.load_weights('MLP.weights.best.hdf5')
model.evaluate(x_test,y_test)

