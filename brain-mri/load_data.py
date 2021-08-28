# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 13:35:40 2019

@author: Jacob
"""


import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt

import keras
from keras.models import Sequential
from keras.optimizers import Adam, SGD
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization

import random
import urllib.request
from pathlib import Path

# %%

# DATA VISUALISATION

def show_slices(slices):
   """ Function to display row of image slices """
   fig, ax = plt.subplots(1, len(slices))
   for axis, slice in zip(ax.flatten(), slices):
        axis.imshow(slice.T, cmap="gray", origin="lower")
        
# %%

# Project path
root = Path("C:/Users/Jacob/Downloads/bpdata")
root2 = root / 'data'

# %%

# DOWNLOADING DATA

download_participants = False
download_mris = False

# Download participant class labels
# Class is participants_df['diagnosis']
# key is participants_df['participant_id']
if download_participants:
    url = "https://openneuro.org/crn/datasets/ds000030/snapshots/00016/files/participants.tsv"
    file = urllib.request.URLopener()
    filename = url.split('/')[-1]
    file.retrieve(url, root / filename)
        
# %%

# Load classes (diagnosis)
participants = root / "participants.tsv"
participants_df = pd.read_csv(participants, sep='\t')
id_diagnosis_dict = participants_df.set_index('participant_id')['diagnosis'].to_dict()

# %%


# Download features (patient rest-state fMRI)
if download_mris:
    for sub in participants_df['participant_id']:
        url = f"https://openneuro.org/crn/datasets/ds000030/snapshots/00016/files/{sub}:func:{sub}_task-rest_bold.nii.gz"
        filename = url.split(':')[-1]
        file = urllib.request.URLopener()
        file.retrieve(url, root2 / filename)
    
    # Deleting missing files
    for id in ('10193','10948','11082','70002'):
        os.remove(root2 / f'sub-{id}_task-rest_bold.nii.gz')


# %%

# Load feature data (MRI files)
        
def get_id(filename):
    """Returns the subject id from the filename of their mri file."""
    return filename.split('/')[-1].split('_')[0]

images = [filename.as_posix() for filename in root2.iterdir()]
images = [(get_id(filename), nib.load(filename)) for filename in images if get_id(filename) in participants_df['participant_id'].to_list()]

j = 6
# Take every jth time stamp
images = [(filename, image.slicer[:,:,:,i].get_fdata()) for filename, image in images for i in range(0,image.shape[-1],j)]

#X = np.array([img[1] for img in images])
#Y = np.array([id_diagnosis_dict[img[0]] for img in images])

#np.save(root / 'fmri.npy', X)
#np.save(root / 'labels.npy', Y)

# %%

# Splitting testing / training
# Need to split on per patient basis, so as not to have bias from different 
# time stamps from same patient in both sets

#random.seed(0)

percent = 0.8

mask1 = participants_df['diagnosis'].isin(['CONTROL','SCHZ','BIPOLAR'])
n = int(percent*sum(mask1))
train_patients = random.sample(participants_df['participant_id'][mask1].to_list(), n)

mask2 = ~participants_df['participant_id'].isin(train_patients) & mask1
test_patients = participants_df[mask2]['participant_id'].to_list()

X_train = np.array([img[1] for img in images if img[0] in train_patients])
X_test = np.array([img[1] for img in images if img[0] in test_patients])

# %%

# Integerify classes
class_dict = {'CONTROL':0, 'SCHZ':1, 'BIPOLAR':2, 'ADHD':3}

# Grabbing train/test labels
Y_train = np.array([class_dict[id_diagnosis_dict[img[0]]] for img in images if img[0] in train_patients])
Y_test = np.array([class_dict[id_diagnosis_dict[img[0]]] for img in images if img[0] in test_patients])

simple = True

if simple:
    id_diagnosis_dict_simple = {}
    # Create simply binary class - healthy/non-healthy
    for key, value in id_diagnosis_dict.items():
        if value == 'CONTROL':
            id_diagnosis_dict_simple[key] = 0
        elif value == 'ADHD':
            id_diagnosis_dict_simple[key] = 2
        elif value in ('SCHZ', 'BIPOLAR'):
            id_diagnosis_dict_simple[key] = 1
    
    Y_train = np.array([id_diagnosis_dict_simple[img[0]] for img in images if img[0] in train_patients])
    Y_test = np.array([id_diagnosis_dict_simple[img[0]] for img in images if img[0] in test_patients])


# %%

# Params
nb_classes = len(set(Y_train))
epochs = 10
batch_size = 128

print(f'x_train shape: {X_train.shape}')
print(f'{X_train.shape[0]} train samples')

# %%

# One hot encode class vectors
y_train = keras.utils.to_categorical(Y_train, nb_classes)
y_test = keras.utils.to_categorical(Y_test, nb_classes)

# %%

# initiate RMSprop optimizer
# optimizer
learning_rate = 1e-5
adam = Adam(lr=learning_rate)
sgd = SGD(lr=learning_rate)
rms = keras.optimizers.RMSprop(learning_rate=0.0001, decay=1e-6)


# Model 2

def load_model2(nb_classes=1000,input_shape=None):
    model = Sequential()
    model.add(Conv2D(32,(5,5),padding='same',input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    

    model.add(Conv2D(32,(5,5),padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Convolution2D(64,(3,3),padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    
    model.add(Conv2D(64,(3,3),padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(64,(3,3),padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    
    model.add(Conv2D(96,(3,3),padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))    
    model.add(Conv2D(96,(3,3),padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    

    model.add(Conv2D(192,(3,3),padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(192,(3,3),padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    model.add(Flatten())
    
    model.add(Dense(units=4096,input_dim=2*2*192))
    model.add(Activation('relu'))
    #model.add(Dropout(0.4)) # for first level
    model.add(Dropout(0.4)) # for sec level
    
    model.add(Dense(units=4096,input_dim=4096))
    model.add(Activation('relu'))
    #model.add(Dropout(0.4)) # for first level
    model.add(Dropout(0.4)) # for sec level
    
    model.add(Dense(units=nb_classes,input_dim=4096))
    model.add(Activation('softmax'))
    
    return model

model = load_model2(nb_classes=nb_classes, input_shape=X_train.shape[1:])

# Train model using RMSprop
model.compile(loss='categorical_crossentropy', optimizer=rms, metrics=['accuracy'])


# %%

# Fit model to data
refit_model = True

if refit_model2:
    model.fit(X_train, 
              y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(X_test, y_test),
              shuffle=True)


# %%

import sklearn.metrics as metrics
import seaborn as sns

confusion_matrix = metrics.confusion_matrix(y_true=Y_test, y_pred=np.argmax(model.predict(X_test), axis=1))

sns.heatmap(confusion_matrix, square=True, annot=True, cmap="YlGnBu")
plt.xlabel('predicted value')
plt.ylabel('true value')
