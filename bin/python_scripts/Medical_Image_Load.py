#########################################################
# Graham Schloesser code data pipline
# Reads in a file of MRI or CT images
# Preprocesses the data and retruns a TensorFlow dataset
# Last edited: 08/19/2021
#########################################################

import tensorflow as tf
import numpy as np
import nibabel as nib
import os

AUTOTUNE = tf.data.experimental.AUTOTUNE
BUFFER_SIZE = 50
BATCH_SIZE = 1
IMG_WIDTH = 512
IMG_HEIGHT = 512
DEPTH = 232

# Normalizing MRI values from [1-,1]
def normalize_MRI(image):
    image = tf.cast(image, tf.float32)
    image = (image / (tf.math.reduce_max(image))*2) - 1
    return image

# Normalizing CT values from [1-,1]
def normalize_CT(image):
    image = tf.cast(image, tf.float32)
    image = image + abs(tf.math.reduce_min(image))
    image = (image / (tf.math.reduce_max(image))*2) - 1
    return image

# Function to call all preprocessing stpes. Added for scalability
def preprocess_MRI(image):
    return normalize_MRI(image)

# Function to call all preprocessing stpes. Added for scalability
def preprocess_CT(image):
    return normalize_CT(image)

#Casting function
def do_nothing(image):
    return tf.cast(image, tf.float32)

"""
pipe_MRI_train takes all MRI data preprocesses it and prepares it for training

:peram path: path of folder that holds MRI data
:peram type: type of data being processed
    0: MRI
    1: CT

:returns tf_data: a tensorflow dataset ready for training
"""
def pipe_train(path,type):
    sample_limit = 0                    #counts samples that have been processed
    for filename in os.listdir(path):   #loops through every file in path
        if filename.endswith('.gz'):    #confirms correct file type
          sample = nib.load(path + '/' + filename).get_fdata()
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

          sample_limit = sample_limit + 1

    train_data = tf.data.Dataset.from_tensor_slices(combined_array)
    #creates tensorflow dataset from array

    tf_data = train_data.map(do_nothing, num_parallel_calls=AUTOTUNE).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    #maps the data for training

    return tf_data
