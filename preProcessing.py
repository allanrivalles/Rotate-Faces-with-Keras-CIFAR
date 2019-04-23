# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 09:58:13 2019

@author: arsf
"""
import os
import shutil
import pandas as pd
from tqdm import tqdm

def main():

    ground_truth = pd.read_csv(os.path.join('train.truth.csv'), sep=',',index_col = None,error_bad_lines=True,dtype=str)

    train_dir = 'train'
    test_dir = 'test'
    val_dir = 'valid'

    val_set = 10 #Percent

    if not(os.path.exists(os.path.join('TensorBoard'))):
        os.makedirs(os.path.join('TensorBoard'))

    if not(os.path.exists(os.path.join('Corrected_Images'))):
        os.makedirs(os.path.join('Corrected_Images'))
    #Train >>>>>>>>>

    #Checking for the balance of classes
    ground_truth['label'].value_counts()
    #The samples are well distributed among classes

    #Create labels as subdirectories
    labels = ground_truth['label'].unique()

    for label in labels:
        if not(os.path.exists(os.path.join(train_dir,label))):
            os.makedirs(os.path.join(train_dir,label))    

    #Distibute files to their respective subdirectories according to their labels read from ground_truth
    for files in tqdm(os.listdir(train_dir)): 
        if not(files in labels): #Folders
            its_label = ground_truth.loc[ground_truth['fn'] == files, 'label'].iloc[0]
            shutil.move(os.path.join(train_dir,files), os.path.join(train_dir,its_label))

    #Validation >>>>>>>>>

    if not(os.path.exists(os.path.join(val_dir))):
        os.makedirs(os.path.join(val_dir))
    if not(os.path.exists(os.path.join(val_dir))):
        os.makedirs(os.path.join(val_dir))

    #Create labels as subdirectories    
    for label in labels:
        if not(os.path.exists(os.path.join(val_dir,label))):
            os.makedirs(os.path.join(val_dir,label)) 
    
    for label in tqdm(labels):       
        q_label = ground_truth.loc[ground_truth['label'] == label].shape[0]
        count = 0
        for files in os.listdir(os.path.join(train_dir,label)):
            if count <= q_label//val_set:
                shutil.move(os.path.join(train_dir,label,files), os.path.join(val_dir,label))
                count  = count + 1
    
    #Test >>>>>>>>>
    
    if not(os.path.exists(os.path.join(test_dir,'Test_folder'))):
        os.makedirs(os.path.join(test_dir,'Test_folder'))
    
    #Moving the files in test folder to a folder called Test_folder inside test folder (necessary to use Keras's ImageDataGenerator)
    for files in tqdm(os.listdir(test_dir)): 
        if not(files == 'Test_folder'): #Folders            
            shutil.move(os.path.join(test_dir,files), os.path.join(test_dir,'Test_folder'))

if __name__ == '__main__':
    main()
    
