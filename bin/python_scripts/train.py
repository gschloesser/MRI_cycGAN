#########################################################
# Graham Schloesser Model Test
# Reads in a MRI and CT files and CycleGAN Model
# Trains the model and saves the weights
# This code was developed by me
# Last edited: 09/13/2021
#
# To adapt this file:
#   change path_MRI and path_CT
#       These paths should be folders containing
#       individual scans.
#   change the checkpoint_path
#   adjust epochs
#
#########################################################

import Medical_Image_Load as MIL
import Keras_Cycle_GAN as GAN

import tensorflow as tf
import numpy as np
import nibabel as nib
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_datasets as tfds
import tensorflow_addons as tfa
import matplotlib.pyplot as plt

import os
import time

path_MRI = '../../data/MRI_Data'
path_CT = '../../data/CT_Data'
checkpoint_path = "/content/drive/MyDrive/GAN_IM/checkpoints/Keras_Test"

input_img_size = (512, 512, 3)

train_MRI = MIL.pipe_train(path_MRI,0)
train_CT = MIL.pipe_train(path_CT,1)

print("Images loaded")

# Loss function for evaluating adversarial loss
adv_loss_fn = keras.losses.MeanSquaredError()

gen_G = GAN.get_resnet_generator(name="generator_G")
gen_F = GAN.get_resnet_generator(name="generator_F")

# Get the discriminators
disc_X = GAN.get_discriminator(name="discriminator_X")
disc_Y = GAN.get_discriminator(name="discriminator_Y")

# Define the loss function for the generators
def generator_loss_fn(fake):
    fake_loss = adv_loss_fn(tf.ones_like(fake), fake)
    return fake_loss

# Define the loss function for the discriminators
def discriminator_loss_fn(real, fake):
    real_loss = adv_loss_fn(tf.ones_like(real), real)
    fake_loss = adv_loss_fn(tf.zeros_like(fake), fake)
    return (real_loss + fake_loss) * 0.5


# Create cycle gan model
cycle_gan_model = GAN.CycleGan(
    generator_G=gen_G, generator_F=gen_F, discriminator_X=disc_X, discriminator_Y=disc_Y
)

# Compile the model

cycle_gan_model.compile(
    gen_G_optimizer=keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),
    gen_F_optimizer=keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),
    disc_X_optimizer=keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),
    disc_Y_optimizer=keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),
    gen_loss_fn=generator_loss_fn,
    disc_loss_fn=discriminator_loss_fn,
)
# Callbacks (not used)
plotter = GAN.GANMonitor()
checkpoint_filepath = "./model_checkpoints/cyclegan_checkpoints.{epoch:03d}"
model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath
)

#Train the model
cycle_gan_model.fit(
    tf.data.Dataset.zip((train_MRI, train_CT)),
    epochs=200,
)


checkpoint_path = '../../checkpoints'
gen_model = cycle_gan_model.gen_G
gen_model.save(checkpoint_path)
