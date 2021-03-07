#!/usr/bin/env python
# coding: utf-8

# # HW: X-ray images classification
# --------------------------------------

# Before you begin, open Mobaxterm and connect to triton with the user and password you were give with. Activate the environment `2ndPaper` and then type the command `pip install scikit-image`.

# In this assignment you will be dealing with classification of 32X32 X-ray images of the chest. The image can be classified into one of four options: lungs (l), clavicles (c), and heart (h) and background (b). Even though those labels are dependent, we will treat this task as multiclass and not as multilabel. The dataset for this assignment is located on a shared folder on triton (`/MLdata/MLcourse/X_ray/'`).

# In[1]:


import os
import numpy as np
from tensorflow.keras.layers import Dense, MaxPool2D, Conv2D, Dropout
from tensorflow.keras.layers import Flatten, InputLayer
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import *

from tensorflow.keras.initializers import Constant
from tensorflow.keras.datasets import fashion_mnist
import tensorflow.keras.backend as K
from tensorflow.keras import regularizers
from tensorflow import keras
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import *
from skimage.io import imread

from skimage.transform import rescale, resize, downscale_local_mean
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]="2"


# In[2]:


import tensorflow as tf
config = tf.compat.v1.ConfigProto(gpu_options =
                         tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8)
# device_count = {'GPU': 1}
)
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)


# In[3]:


def preprocess(datapath):
    # This part reads the images
    classes = ['b','c','l','h']
    imagelist = [fn for fn in os.listdir(datapath)]
    N = len(imagelist)
    num_classes = len(classes)
    images = np.zeros((N, 32, 32, 1))
    Y = np.zeros((N,num_classes))
    ii=0
    for fn in imagelist:

        src = imread(os.path.join(datapath, fn),1)
        img = resize(src,(32,32),order = 3)
        
        images[ii,:,:,0] = img
        cc = -1
        for cl in range(len(classes)):
            if fn[-5] == classes[cl]:
                cc = cl
        Y[ii,cc]=1
        ii += 1

    BaseImages = images
    BaseY = Y
    return BaseImages, BaseY


# In[4]:


def preprocess_train_and_val(datapath):
    # This part reads the images
    classes = ['b','c','l','h']
    imagelist = [fn for fn in os.listdir(datapath)]
    N = len(imagelist)
    num_classes = len(classes)
    images = np.zeros((N, 32, 32, 1))
    Y = np.zeros((N,num_classes))
    ii=0
    for fn in imagelist:

        images[ii,:,:,0] = imread(os.path.join(datapath, fn),1)
        cc = -1
        for cl in range(len(classes)):
            if fn[-5] == classes[cl]:
                cc = cl
        Y[ii,cc]=1
        ii += 1

    return images, Y


# In[5]:


#Loading the data for training and validation:
src_data = '/MLdata/MLcourse/X_ray/'
#src_data = 'C:\\Users\\ofeka\\Desktop\\Projects\\Machine Learning\\Homework\\HW4\\X_ray\\'
train_path = src_data + 'train'
val_path = src_data + 'validation'
test_path = src_data + 'test'
BaseX_train , BaseY_train = preprocess_train_and_val(train_path)
BaseX_val , BaseY_val = preprocess_train_and_val(val_path)
X_test, Y_test = preprocess(test_path)


# In[6]:


keras.backend.clear_session()


# In[7]:


# BLOCK FOR TESTING STUFF:

print(BaseX_train.shape)
print(BaseX_train.flatten().shape)
print(BaseY_train.shape)
print(BaseX_val.shape)
print(BaseY_val.shape)
print(X_test.shape)
print(Y_test.shape)


# ### PART 1: Fully connected layers 
# --------------------------------------

# ---
# <span style="color:red">***Task 1:***</span> *NN with fully connected layers. 
# 
# Elaborate a NN with 2 hidden fully connected layers with 300, 150 neurons and 4 neurons for classification. Use ReLU activation functions for the hidden layers and He_normal for initialization. Don't forget to flatten your image before feedforward to the first dense layer. Name the model `model_relu`.*
# 
# ---

# In[64]:


#--------------------------Impelment your code here:-------------------------------------
n_filters_start = 300
n_filters_finish = 4
len_sub_window = 10
dropout = 0.2
he_normal_initializer = tf.keras.initializers.he_normal()
model_relu = Sequential()
model_relu.add(Flatten(input_shape = (32,32,1))) #input_shape defines the shape of the input: 32X32 is the image size, while 3 is the number of dimensions (RGB hence 3)
model_relu.add(Dense(n_filters_start, activation='relu',kernel_initializer = he_normal_initializer)) 
model_relu.add(Dropout(dropout))
model_relu.add(Dense(n_filters_start/2,activation='relu',kernel_initializer = he_normal_initializer))
model_relu.add(Dropout(dropout))
model_relu.add(Dense(n_filters_finish,activation='softmax'))


# 2nd try model
# model_relu_d = Sequential(name="model_relu")
# model_relu_d.add(Dense(300, input_shape=(32 ** 2,), kernel_initializer="he_normal"))
# model_relu_d.add(Activation('relu', name='ReLU_1'))
# model_relu_d.add(Dropout(0.2))

# model_relu_d.add(Dense(150, kernel_initializer="he_normal"))
# model_relu_d.add(Activation('relu', name='ReLU_2'))
# model_relu_d.add(Dropout(0.2))

# model_relu_d.add(Dense(4))
# model_relu_d.add(Activation('softmax'))
#----------------------------------------------------------------------------------------


# In[65]:


model_relu.summary()


# In[66]:


#Inputs: 
input_shape = (32,32,1)
learn_rate = 1e-5
decay = 0
batch_size = 64
epochs = 25

#Define your optimizar parameters:
AdamOpt = Adam(lr=learn_rate,decay=decay)


# 28/2/2021 - Not finished: Still need to define paarameters


# Compile the model with the optimizer above, accuracy metric and adequate loss for multiclass task. Train your model on the training set and evaluate the model on the testing set. Print the accuracy and loss over the testing set.

# In[67]:


#--------------------------Impelment your code here:-------------------------------------
model_relu.compile(optimizer = AdamOpt, metrics=['accuracy'], loss='categorical_crossentropy')
model_relu.fit(BaseX_train, BaseY_train, epochs = epochs, batch_size = batch_size, validation_data=(BaseX_val,BaseY_val))

# This crossentropy loss function is used when there are two or more label classes. We expect labels to be provided in a one_hot representation. 
# If labels were  provided as integers, then SparseCategoricalCrossentropy loss should be used. There should be # classes floating point values per feature.
#----------------------------------------------------------------------------------------


# In[68]:


model_relu.evaluate(X_test, Y_test)

# Saving the model:
if not("results" in os.listdir()):
    os.mkdir("results")
save_dir = "results/"

#model_relu:
model_name = "model_relu.h5"
model_path = os.path.join(save_dir, model_name)
model_relu.save(model_path)
print('Saved trained model at %s ' % model_path)


# ---
# <span style="color:red">***Task 2:***</span> *Activation functions.* 
# 
# Change the activation functions to LeakyRelu or tanh or sigmoid. Name the new model `new_a_model`. Explain how it can affect the model.*
# 
# ---

# In[13]:


#--------------------------Impelment your code here:-------------------------------------
#import keras.layers.advanced_activations as advanced_activations
leaky_relu = LeakyReLU(alpha=0.1)


new_a_model = Sequential()
new_a_model.add(Flatten(input_shape = (32,32,1))) #input_shape defines the shape of the input: 32X32 is the image size, while 3 is the number of dimensions (RGB hence 3)
new_a_model.add(leaky_relu)
new_a_model.add(Dense(n_filters_start,kernel_initializer = he_normal_initializer)) 
new_a_model.add(leaky_relu)
new_a_model.add(Dropout(dropout))
new_a_model.add(Dense(n_filters_start/2))
new_a_model.add(leaky_relu)
new_a_model.add(Dropout(dropout))
new_a_model.add(Dense(n_filters_finish)) # 28/2/2021: Might need to be deleted.
new_a_model.add(Dense(n_filters_finish,activation='softmax'))

new_a_model_25 = new_a_model
new_a_model_40 = new_a_model
#----------------------------------------------------------------------------------------


# In[14]:


new_a_model.summary()


# ---
# <span style="color:red">***Task 3:***</span> *Number of epochs.* 
# 
# Train the new model using 25 and 40 epochs. What difference does it makes in term of performance? Remember to save the compiled model for having initialized weights for every run as we did in tutorial 12. Evaluate each trained model on the test set*
# 
# ---

# In[30]:


#Inputs: 
input_shape = (32,32,1)
learn_rate = 1e-5
decay = 0
batch_size = 64
epochs = 25

# 28/02/2021: finish the following:
#Defining the optimizar parameters:
AdamOpt = Adam(lr=learn_rate,decay=decay)


# In[31]:


#--------------------------Impelment your code here:-------------------------------------
new_a_model_25.compile(optimizer = AdamOpt, metrics=['accuracy'], loss='categorical_crossentropy')
new_a_model_25.fit(BaseX_train, BaseY_train, epochs = epochs, batch_size = batch_size, validation_data=(BaseX_val,BaseY_val))
#-----------------------------------------------------------------------------------------


# In[36]:


#Inputs: 
input_shape = (32,32,1)
learn_rate = 1e-5
decay = 0
batch_size = 64
epochs = 40

#Defining the optimizar parameters:
AdamOpt = Adam(lr=learn_rate,decay=decay)


# ### Evaluation results 25 epochs:

# In[37]:


new_a_model_25.evaluate(X_test, Y_test)

# Saving the model:
#new_a_model - 25 epochs:
model_name = "new_a_model_25.h5"
model_path = os.path.join(save_dir, model_name)
new_a_model_25.save(model_path)
print('Saved trained model at %s ' % model_path)


# In[38]:


#--------------------------Impelment your code here:-------------------------------------

new_a_model_40.compile(optimizer = AdamOpt, metrics=['accuracy'], loss='categorical_crossentropy')
new_a_model_40.fit(BaseX_train, BaseY_train, epochs = epochs, batch_size = batch_size, validation_data=(BaseX_val,BaseY_val))

#-----------------------------------------------------------------------------------------


# ### Evaluation results 40 epochs:

# In[39]:


new_a_model_40.evaluate(X_test, Y_test)


# ### Saving the models:

# In[40]:


# Saving the model:
#new_a_model - 40 epochs:
model_name = "new_a_model_40.h5"
model_path = os.path.join(save_dir, model_name)
new_a_model_40.save(model_path)
print('Saved trained model at %s ' % model_path)


# ---
# <span style="color:red">***Task 4:***</span> *Mini-batches.* 
# 
# Build the `model_relu` again and run it with a batch size of 32 instead of 64. What are the advantages of the mini-batch vs. SGD?*
# 
# ---

# In[41]:


keras.backend.clear_session()


# In[42]:


#--------------------------Impelment your code here:-------------------------------------
# original 64 batch-size model:
model_relu_32b = Sequential()
model_relu_32b.add(Flatten(input_shape = (32,32,1))) #input_shape defines the shape of the input: 32X32 is the image size, while 3 is the number of dimensions (RGB hence 3)
model_relu_32b.add(Dense(n_filters_start, activation='relu',kernel_initializer = he_normal_initializer)) 
model_relu_32b.add(Dropout(dropout))
model_relu_32b.add(Dense(n_filters_start/2,activation='relu'))
model_relu_32b.add(Dropout(dropout))
model_relu_32b.add(Dense(n_filters_finish,activation='softmax'))

#----------------------------------------------------------------------------------------


# In[43]:


batch_size = 32
epochs = 50

#Define your optimizar parameters:
AdamOpt = Adam(lr=learn_rate,decay=decay)


# In[44]:


#--------------------------Impelment your code here:-------------------------------------
model_relu_32b.compile(optimizer = AdamOpt, metrics=['accuracy'], loss='categorical_crossentropy')
model_relu_32b.fit(BaseX_train, BaseY_train, epochs = epochs, batch_size = batch_size, validation_data=(BaseX_val,BaseY_val))
#----------------------------------------------------------------------------------------


# In[50]:


model_relu_32b.evaluate(X_test, Y_test)


# ---
# <span style="color:red">***Task 4:***</span> *Batch normalization.* 
# 
# Build the `new_a_model` again and add batch normalization layers. How does it impact your results?*
# 
# ---

# In[46]:


keras.backend.clear_session()


# In[47]:


#--------------------------Impelment your code here:-------------------------------------
# 28/02/2021: Make sure that BatchNormaliztion() does not need any parameters:
new_a_model_batch = Sequential()
new_a_model_batch.add(Flatten(input_shape = (32,32,1))) #input_shape defines the shape of the input: 32X32 is the image size, while 3 is the number of dimensions (RGB hence 3)
new_a_model_batch.add(leaky_relu)
new_a_model_batch.add(BatchNormalization())
new_a_model_batch.add(Dense(n_filters_start,kernel_initializer = he_normal_initializer)) 
new_a_model_batch.add(leaky_relu)
new_a_model_batch.add(BatchNormalization())
new_a_model_batch.add(Dropout(dropout))
new_a_model_batch.add(BatchNormalization())
new_a_model_batch.add(Dense(n_filters_start/2))
new_a_model_batch.add(leaky_relu)
new_a_model_batch.add(BatchNormalization())
new_a_model_batch.add(Dropout(dropout))
new_a_model_batch.add(BatchNormalization())
new_a_model_batch.add(Dense(n_filters_finish))
new_a_model_batch.add(BatchNormalization())
new_a_model_batch.add(Dense(n_filters_finish,activation='softmax'))


#---------------------------------------------------------------------------------------


# In[48]:


batch_size = 32
epochs = 50

#Define your optimizar parameters:
AdamOpt = Adam(lr=learn_rate,decay=decay)
#Compile the network: 



# In[49]:


#Preforming the training by using fit 
#--------------------------Impelment your code here:-------------------------------------
new_a_model_batch.compile(optimizer = AdamOpt, metrics=['accuracy'], loss='categorical_crossentropy')
new_a_model_batch.fit(BaseX_train, BaseY_train, epochs = epochs, batch_size = batch_size, validation_data=(BaseX_val,BaseY_val))
#----------------------------------------------------------------------------------------
# 28/02/2021: add evaluate()!!! and answer the question: "How does it impact your results?"


# In[51]:


new_a_model_batch.evaluate(X_test, Y_test)


# In[52]:


#model_relu - batch = 32:
model_name = "model_relu_32.h5"
model_path = os.path.join(save_dir, model_name)
model_relu_32b.save(model_path)
print('Saved trained model at %s ' % model_path)

#new_a_model - with batch normalization:
model_name = "new_a_model_batch_norm.h5"
model_path = os.path.join(save_dir, model_name)
new_a_model_batch.save(model_path)
print('Saved trained model at %s ' % model_path)


# ### PART 2: Convolutional Neural Network (CNN)
# ------------------------------------------------------------------------------------

# ---
# <span style="color:red">***Task 1:***</span> *2D CNN.* 
# 
# Have a look at the model below and answer the following:
# * How many layers does it have?
# * How many filter in each layer?
# * Would the number of parmaters be similar to a fully connected NN?
# * Is this specific NN performing regularization?
# 
# ---

# In[53]:


def get_net(input_shape,drop,dropRate,reg):
    #Defining the network architecture:
    model = Sequential()
    model.add(Permute((1,2,3),input_shape = input_shape))
    model.add(Conv2D(filters=64, kernel_size=(3,3), padding='same', activation='relu',name='Conv2D_1',kernel_regularizer=regularizers.l2(reg)))
    if drop:
        model.add(Dropout(rate=dropRate))
    model.add(BatchNormalization(axis=1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=128, kernel_size=(3,3), padding='same', activation='relu',name='Conv2D_2',kernel_regularizer=regularizers.l2(reg)))
    if drop:    
        model.add(Dropout(rate=dropRate))
    model.add(BatchNormalization(axis=1))
    model.add(Conv2D(filters=128, kernel_size=(3,3), padding='same', activation='relu',name='Conv2D_3',kernel_regularizer=regularizers.l2(reg)))
    if drop:
        model.add(Dropout(rate=dropRate))
    model.add(BatchNormalization(axis=1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding='same', activation='relu',name='Conv2D_4',kernel_regularizer=regularizers.l2(reg)))
    if drop:
        model.add(Dropout(rate=dropRate))
    model.add(BatchNormalization(axis=1))
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding='same', activation='relu',name='Conv2D_5',kernel_regularizer=regularizers.l2(reg)))
    if drop:
        model.add(Dropout(rate=dropRate))
    model.add(BatchNormalization(axis=1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    #Fully connected network tail:      
    model.add(Dense(512, activation='elu',name='FCN_1')) 
    if drop:
        model.add(Dropout(rate=dropRate))
    model.add(Dense(128, activation='elu',name='FCN_2'))
    model.add(Dense(4, activation= 'softmax',name='FCN_3'))
    model.summary()
    return model


# In[54]:


input_shape = (32,32,1)
learn_rate = 1e-5
decay = 1e-03
batch_size = 64
epochs = 25
drop = True
dropRate = 0.3
reg = 1e-2
NNet = get_net(input_shape,drop,dropRate,reg)


# In[70]:


# 28/2/2021: Why get_net is used twice with same parameters?
NNet=get_net(input_shape,drop,dropRate,reg)


# In[55]:


from tensorflow.keras.optimizers import *
import os
from tensorflow.keras.callbacks import *

#Defining the optimizar parameters:
AdamOpt = Adam(lr=learn_rate,decay=decay)

#Compile the network: 
NNet.compile(optimizer=AdamOpt, metrics=['acc'], loss='categorical_crossentropy')

#Saving checkpoints during training:
Checkpath = os.getcwd()
Checkp = ModelCheckpoint(Checkpath, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=True, save_freq=1)


# In[56]:


#Preforming the training by using fit 
# IMPORTANT NOTE: This will take a few minutes!
h = NNet.fit(x=BaseX_train, y=BaseY_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_split=0, validation_data = (BaseX_val, BaseY_val), shuffle=True)
#NNet.save(model_fn)


# In[ ]:


# NNet.load_weights('Weights_1.h5')


# In[57]:


#new_a_model - with batch normalization:
model_name = "NNet.h5"
model_path = os.path.join(save_dir, model_name)
NNet.save(model_path)
print('Saved trained model at %s ' % model_path)


# In[58]:


results = NNet.evaluate(X_test,Y_test)
print('test loss, test acc:', results)


# ---
# <span style="color:red">***Task 2:***</span> *Number of filters* 
# 
# Rebuild the function `get_net` to have as an input argument a list of number of filters in each layers, i.e. for the CNN defined above the input should have been `[64, 128, 128, 256, 256]`. Now train the model with the number of filters reduced by half. What were the results.
# 
# ---

# In[59]:


#--------------------------Impelment your code here:-------------------------------------
def get_net(input_shape,drop,dropRate,reg, filter_num1, filter_num2, filter_num3, filter_num4, filter_num5):
    #Defining the network architecture:
    model = Sequential()
    model.add(Permute((1,2,3),input_shape = input_shape))
    model.add(Conv2D(filters=filter_num1, kernel_size=(3,3), padding='same', activation='relu',name='Conv2D_1',kernel_regularizer=regularizers.l2(reg)))
    if drop:
        model.add(Dropout(rate=dropRate))
    model.add(BatchNormalization(axis=1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=filter_num2, kernel_size=(3,3), padding='same', activation='relu',name='Conv2D_2',kernel_regularizer=regularizers.l2(reg)))
    if drop:    
        model.add(Dropout(rate=dropRate))
    model.add(BatchNormalization(axis=1))
    model.add(Conv2D(filters=filter_num3, kernel_size=(3,3), padding='same', activation='relu',name='Conv2D_3',kernel_regularizer=regularizers.l2(reg)))
    if drop:
        model.add(Dropout(rate=dropRate))
    model.add(BatchNormalization(axis=1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=filter_num4, kernel_size=(3,3), padding='same', activation='relu',name='Conv2D_4',kernel_regularizer=regularizers.l2(reg)))
    if drop:
        model.add(Dropout(rate=dropRate))
    model.add(BatchNormalization(axis=1))
    model.add(Conv2D(filters=filter_num5, kernel_size=(3,3), padding='same', activation='relu',name='Conv2D_5',kernel_regularizer=regularizers.l2(reg)))
    if drop:
        model.add(Dropout(rate=dropRate))
    model.add(BatchNormalization(axis=1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    #Fully connected network tail:      
    model.add(Dense(512, activation='elu',name='FCN_1')) 
    if drop:
        model.add(Dropout(rate=dropRate))
    model.add(Dense(128, activation='elu',name='FCN_2'))
    model.add(Dense(4, activation= 'softmax',name='FCN_3'))
    model.summary()
    return model
#----------------------------------------------------------------------------------------


# In[60]:


filter_n1 = 32
filter_n2 = 64
filter_n3 = 64
filter_n4 = 128
filter_n5 = 128
# 28/02/2021: Change the input to a "list" instead of 5 int?
NNet_half = get_net(input_shape,drop,dropRate,reg, filter_n1, filter_n2, filter_n3, filter_n4, filter_n5)


# In[61]:


#Defining the optimizar parameters:
AdamOpt = Adam(lr=learn_rate,decay=decay)

#Compile the network: 
NNet_half.compile(optimizer=AdamOpt, metrics=['acc'], loss='categorical_crossentropy')

#Saving checkpoints during training:
Checkpath = os.getcwd()
Checkp = ModelCheckpoint(Checkpath, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=True, save_freq=1)

#Preforming the training by using fit 
# IMPORTANT NOTE: This will take a few minutes!
h = NNet_half.fit(x=BaseX_train, y=BaseY_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_split=0, validation_data = (BaseX_val, BaseY_val), shuffle=True)
#NNet_half.save(model_fn)


# In[62]:


#new_a_model - with batch normalization:
model_name = "NNet_half.h5"
model_path = os.path.join(save_dir, model_name)
NNet_half.save(model_path)
print('Saved trained model at %s ' % model_path)


# In[63]:


results = NNet_half.evaluate(X_test,Y_test)
print('test loss, test acc:', results)


# That's all folks! See you :)
