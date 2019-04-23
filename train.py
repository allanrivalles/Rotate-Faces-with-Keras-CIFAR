# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 14:08:36 2019

@author: arsf
"""
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import rmsprop
from keras.callbacks import TensorBoard
from time import time
import os

def main():
    
    batch_size = 100
    num_classes = 4
    epochs = 13
    
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
    
    valid_generator = train_datagen.flow_from_directory(
        directory=os.path.join('valid'),
        target_size=(64, 64),
        color_mode="rgb",
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True,
        seed=42)
    
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

    tensorboard = TensorBoard(log_dir="TensorBoard/tensor_board_rotFaces_"+str(time()))

    #Ativar Tensor Board -> tensorboard --logdir="TensorBoard"/ --host localhost --port 8088

    STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
    STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size
    model.fit_generator(generator=train_generator, verbose=1,
                        steps_per_epoch=STEP_SIZE_TRAIN,
                        validation_data=valid_generator,
                        validation_steps=STEP_SIZE_VALID,
                        callbacks=[tensorboard],                        
                        epochs=epochs
    )
   
    model.save_weights('Rot_Faces_Conv_Net.h5')    

if __name__ == '__main__':
    main()
    