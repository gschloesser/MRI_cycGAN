#########################################################
# Graham Schloesser Model Test
# Reads in trained weights
# This code was developed by me
# This file tests a saved model and isolates the bone
# structure and calculates the MSE. 
# Last edited: 09/13/2021
#
# To adapt this file:
#   change path_MRI and path_CT
#       These paths should be folders containing
#       individual scans.
#   change the checkpoint_path
#
#########################################################

import tensorflow as tf
import numpy as np
import nibabel as nib
from tensorflow import keras
import os
import matplotlib.pyplot as plt
from google.colab import drive
drive.mount('/content/drive')

checkpoint_path = '../../checkpoints'

model = keras.models.load_model(checkpoint_path)

example_filename_CT = '../../data/CT_Data/scan1'
example_filename_MRI = '../../data/MRI_Data/scan1'
AUTOTUNE = tf.data.AUTOTUNE
OUTPUT_CHANNELS = 3
BUFFER_SIZE = 250
BATCH_SIZE = 1

def normalize_MRI(image):
    image = tf.cast(image, tf.float32)
    image = (image / (tf.math.reduce_max(image))*2) - 1
    #image = tf.image.resize(image,[256,256])
    return image

def normalize_CT(image):
    image = tf.cast(image, tf.float32)
    image = image + abs(tf.math.reduce_min(image))
    image = (image / (tf.math.reduce_max(image))*2) - 1
    #image = tf.image.resize(image,[256,256])
    return image

def preprocess_MRI(image):
    #image = random_jitter(image)
    image = normalize_MRI(image)
    return image
def preprocess_CT(image):
    #image = random_jitter(image)
    image = normalize_CT(image)
    return image

def do_nothing(image):
    return tf.cast(image, tf.float32)

def pipe_test(path,type):
    sample_limit = 0
    thresh = 1
    if path.endswith('.gz'):
      sample = nib.load(path).get_fdata()
      samp_shape = np.shape(sample)
      temp_array = np.zeros((samp_shape[2],samp_shape[0],samp_shape[1],3))
      #loads data and creates temparary array to store data
      for i in range(samp_shape[2]-2):
          if type == 0:     #for MRI images
              temp_array[i,:,:,0:2] = preprocess_MRI(sample[:,:,i:i+2])
          if type == 1:     #for CT images
              temp_array[i,:,:,0:2] = preprocess_CT(sample[:,:,i:i+2])

      if sample_limit == 0:
          combined_array = temp_array
      else:
          combined_array = np.append(combined_array,temp_array,axis=0)
          #appends the temparary array into array that holds all samples


    train_data = tf.data.Dataset.from_tensor_slices(combined_array)
    #creates tensorflow dataset from array

    tf_data = train_data.map(do_nothing, num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE)
    #maps the data for training

    return tf_data


test_MRI = pipe_test(example_filename_MRI,0)
test_CT = pipe_test(example_filename_CT,1)

i = 0
for image_x, image_y in tf.data.Dataset.zip((test_MRI, test_CT)):
  _, ax = plt.subplots(1, 3,figsize = (18,12))
  pred = model(image_x)[0].numpy()
  prediction = ((pred + 1.0) * 127.5).astype(np.uint8)
  img = ((image_x[0] + 1.0) * 127.5).numpy().astype(np.uint8)
  img1 = ((image_y[0] + 1.0) * 127.5).numpy().astype(np.uint8)

  prediction = prediction[:,:,0]
  img = img[:,:,0]
  img1 = img1[:,:,0]

  pred1 = (prediction > (.5*255)) * prediction
  img1 = (img1 > (.5*255)) * img1

  ax[0].imshow(img,cmap='gray')
  ax[1].imshow(pred1,cmap='gray')
  ax[2].imshow(img1,cmap='gray')

  ax[0].set_title("Input image")
  ax[1].set_title("Translated image")
  ax[2].set_title("True CT image")

  ax[0].axis("off")
  ax[1].axis("off")
  ax[2].axis("off")
  if i == 229:
    break
  i = i + 1
  plt.show()


i = 0
arr = np.array([0])
for image_x, image_y in tf.data.Dataset.zip((test_MRI, test_CT)):
  pred = model(image_x)[0].numpy()
  pred1= ((pred + 1.0) * 0.5)
  img = ((image_x[0] + 1.0) * 0.5).numpy()
  img1 = ((image_y[0] + 1.0) * 0.5).numpy()



  pred1 = (pred1 > (.5*1)) * pred1
  img1 = (img1 > (.5*1)) * img1

  mse = tf.keras.metrics.mean_squared_error(img1, pred1).numpy()
  MSE = np.mean(mse)
  arr = np.append(arr,MSE)

arr1 = arr[5:220]
plt.xlabel("Voxel Depth")
plt.ylabel("Mean Squared Error")
plt.plot(arr1)
