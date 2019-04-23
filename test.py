# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 14:01:40 2019

@author: arsf
"""

from keras.models import load_model
import cv2
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.optimizers import rmsprop
import os
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D

def main():
    
    batch_size = 100
    num_classes = 4
    
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=(64, 64, 3)))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    
    opt = rmsprop(lr=0.0001, decay=1e-6)
    
    model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])
    
    model.load_weights('Rot_Faces_Conv_Net.h5')
    
    
    train_datagen = ImageDataGenerator(
    rescale = 1./255,
    shear_range = 0.2, # random application of shearing
    zoom_range = 0.2,
    horizontal_flip = False) # randomly flipping half of the images horizontally

    train_generator = train_datagen.flow_from_directory(
        directory=os.path.join('train'),
        target_size=(64, 64),
        color_mode="rgb",
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True,
        seed=42)
    
    test_datagen = ImageDataGenerator(rescale = 1./255)
    
    test_generator = test_datagen.flow_from_directory(
        directory=os.path.join('test'),
        target_size=(64, 64),
        color_mode="rgb",
        batch_size=1,
        class_mode=None,
        shuffle=False,
        seed=42
    )
    
    test_generator.reset()
    
    pred=model.predict_generator(test_generator,steps=test_generator.n,verbose=1)
    
    predicted_class_indices=np.argmax(pred,axis=1)
    
    labels = (train_generator.class_indices)
    labels = dict((v,k) for k,v in labels.items())
    predictions = [labels[k] for k in predicted_class_indices]
    
    filenames=test_generator.filenames
    results=pd.DataFrame({"fn":filenames,"label":predictions})
    
    results['fn'] = [i.replace('Test_folder\\','') for i in results['fn']]
    
    results.to_csv("preds.csv",index=False)
    
if __name__ == '__main__':
    main()