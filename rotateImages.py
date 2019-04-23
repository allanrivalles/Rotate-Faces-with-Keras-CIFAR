# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 15:56:47 2019

@author: arsf
"""
from tqdm import tqdm
import os
from PIL import Image
import pandas as pd

def main():    
    
    path = r'test\Test_folder'
    
    predicted = pd.read_csv(os.path.join('preds.csv'), sep=',',index_col = None,error_bad_lines=True,dtype=str)

    for index,row in tqdm(predicted.iterrows()):
        image = Image.open(os.path.join(path,row['fn']))  
        
        row['fn'] = row['fn'].replace('.jpg','')
        
        if row['label'] == 'upright':
            rotated = image
        elif row['label'] == 'upside_down':
            rotated  = image.rotate(180)    
        elif row['label'] == 'rotated_right':
            rotated  = image.rotate(90)    
        elif row['label'] == 'rotated_left':
            rotated  = image.rotate(-90)    
                
        rotated.save(os.path.join('Corrected_Images',row['fn']+'.png'))

if __name__ == '__main__':
    main()