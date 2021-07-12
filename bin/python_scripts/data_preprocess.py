import tensorflow as tf
import numpy as np
import nibabel as nib
from tensorflow import keras
from tensorflow.keras import layers
import os
import matplotlib.pyplot as plt

path1 = '/content/drive/My Drive/GAN_IM/MR_Pair'
path2 = '/content/drive/My Drive/GAN_IM/CT_Pair'

IMG_WIDTH = 512
IMG_HEIGHT = 512
DEPTH = 232

def random_crop(image):
    cropped_image = tf.image.random_crop(image, size=[IMG_HEIGHT, IMG_WIDTH])
    return cropped_image

def normalize_MRI(image):
    image = tf.cast(image, tf.float32)
    image = (image / (tf.math.reduce_max(image))*2) - 1
    return image

def normalize_CT(image):
    image = tf.cast(image, tf.float32)
    image = image + abs(tf.math.reduce_min(image))
    image = (image / (tf.math.reduce_max(image))*2) - 1
    return image

def preprocess_MRI(image):
    image = random_jitter(image)
    image = normalize_MRI(image)
    return image
def preprocess_CT(image):
    image = random_jitter(image)
    image = normalize_CT(image)
    return image

def pipe_MRI_train(path):
    j = 0
    for filename in os.listdir(path):
      img = nib.load(path + '/' + filename)
      arr = img.get_fdata()
      sz = np.shape(arr)
      arr1 = np.zeros((sz[2],sz[0],sz[1],3))
      for i in range(DEPTH-2):
          arr1[i,:,:,0] = preprocess_MRI(arr[:,:,i])
          arr1[i,:,:,1] = preprocess_MRI(arr[:,:,i+1])
          arr1[i,:,:,2] = preprocess_MRI(arr[:,:,i+2])
      if j == 0:
          arr2 = arr1
      else:
          arr2 = np.append(arr2,arr1,axis=0)
      j = j + 1
      if j == 3:
        break
      print(np.shape(arr2))
    train_data = tf.data.Dataset.from_tensor_slices(arr2)
    data = train_data.map(do_nothing, num_parallel_calls=AUTOTUNE).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    return data

def pipe_CT_train(path):
    j = 0
    for filename in os.listdir(path):
      img = nib.load(path + '/' + filename)
      arr = img.get_fdata()
      sz = np.shape(arr)
      arr1 = np.zeros((sz[2],sz[0],sz[1],3))
      for i in range(DEPTH-2):
          arr1[i,:,:,0] = preprocess_CT(arr[:,:,i])
          arr1[i,:,:,1] = preprocess_CT(arr[:,:,i+1])
          arr1[i,:,:,2] = preprocess_CT(arr[:,:,i+2])
      if j == 0:
          arr2 = arr1
      else:
          arr2 = np.append(arr2,arr1,axis=0)
      j = j + 1
      if j == 3:
        break
      print(np.shape(arr2))
    train_data = tf.data.Dataset.from_tensor_slices(arr2)
    data = train_data.map(do_nothing, num_parallel_calls=AUTOTUNE).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    return data

def do_nothing(image):
    return tf.cast(image, tf.float32)
