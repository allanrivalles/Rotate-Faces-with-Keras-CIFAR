# Rotate-Faces-with-Keras-CIFAR

This repository is the answer to the challenge of simple computer vision placed in https://gist.github.com/csaftoiu/9fccaf47fd8f96cd378afd8fdd0d63c1

To run the experiment is necessary to perform the following steps:

1) Place the files: PreProcessing.py, train.py, test.py, rotateImages.py, test.rotfaces.zip (from challenge), train.rotfaces.zip (from challenge) in the same directory.

2) Extract the content of the test.rotfaces.zip and train.rotfaces.zip. Two folders (train and test) and the 'train.truth.csv' file should be seen after the extraction.

3) Run the 'PreProcessing.py' script. The content of the test and train folders will be reorganized and there will be created the folders: Corrented_images, TensorBoard and valid.

4) Run the 'train.py' script. It will train a CIFAR convolutional network over the train folder samples and validate the training over the content of the valid folder. At the end of the execution it will be available a Tensor Board file inside the TesnsorBoard folder containing the statistics about the model's training. Those statistics can visualized by starting the tensor board server (localhost). Besides, the model trained weights will be saved as 'Rot_Faces_Conv_Net.h5' file.

5) Run the 'test.py' script. This script will classify the content of the test folder acording to the orientation of the pictures and generate the 'preds.csv' file, which contains the filenames of the test samples along with their respective orientations predicted by the model.

6) Run the script 'eval.py' with the 'truth.csv' file, provided by the challenge owner and the preds.csv generated by the 'test.py' script.

7) Run the 'rotateImages.py', which will adjust the orientation of the test files according to their predicted current orientation and save them in the Corrected_images folder. This folder was zipped and is available in the following link:
https://www.dropbox.com/s/nxtv2rs5t2usvbh/Corrected_Images.zip?dl=0
