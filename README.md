# MRI_cycGAN
UW-Madison ECE697 Sumer Project Codebase
The jupyter notbook contains my progress in model development in Tensorflow.
It has been ran in google colabs with the dataset stored in google drive.
It has also been ran in Euler Clusters to scale up training capability. 

To run the main branch visit the bin/python_scripts folder. Make sure that you have all
libraries availble on your device. Github doesn't have enough storage to store the dataset so,
make sure that you have your own dataset.

I have also provided a Jupyter Notebook that runs in Google Collaborate fairly easily where your
data is stored in your drive. Make sure you use the keras version for the best results. Addionally,
make sure you have access to a GPU training device. 

TO RUN SCRIPT 

Navigate to /bin/python_script directory.

This directory contains all the scripts I made. Keras_Cycle_GAN.py contains the model architecture. Medical_Image_Load.py
provides the code to load medical images and store them into datatypes that can be passed into models. 
train.py trains the cycleGAN model and saves the model weights. test.py tests the saved weights on paired images. 

Make sure that the file paths in test.py and train.py have been adapted for your configuration. Make sure that you
have a folder to save the weights as well. Feel free to change some of the learning perameters such as learning
rates or epochs in the train.py file. 

Run train.py and this will train the model and save the wieghts. Then run test.py to evaluate the model performance. 
