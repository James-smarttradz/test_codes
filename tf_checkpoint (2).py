# DCGAN
# https://www.tensorflow.org/tutorials/generative/dcgan

from keras.models import Sequential
from keras.layers import Conv2D, Dropout, Dense, Flatten, BatchNormalization, LeakyReLU, Reshape, Conv2DTranspose
from keras.losses import BinaryCrossentropy
import matplotlib.pyplot as plt

#pip install -q imageio
#pip install -q git+https://github.com/tensorflow/docs
import glob
import imageio
import numpy as np
import os
import PIL
import time
import tensorflow as tf
from IPython import display
from keras.datasets import mnist
from tensorflow.data import Dataset

# IMPORT MNIST DATASET
(train_images, train_labels), (_, _) = mnist.load_data()
train_images_float=train_images.reshape(train_images.shape[0],28,28,1).astype('float32')
train_images.shape
train_images.dtype
train_images_float.shape
train_images_float.dtype


# SHOW DATASET IMAGES
class_names = ['0', '1', '2', '3','4','5','6','7','8','9']

plt.figure(figsize=(8,8))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images_float[i], cmap=plt.cm.binary)
    # The CIFAR labels happen to be arrays,
    # which is why you need the extra index
    plt.xlabel(class_names[train_labels[i]])

# NORMALISE DATASET IMAGES
train_images_float_norm = (train_images_float - 127.5) / 127.5
train_images_float_norm.min(),train_images_float_norm.max()

BUFFER_SIZE = 60000
BATCH_SIZE = 256

# BATCH AND SHUFFLE THE DATA
# 'from_tensor_slices' - Get the slices of an array in the form of objects
# Refer here - https://www.geeksforgeeks.org/tensorflow-tf-data-dataset-from_tensor_slices/
train_dataset = Dataset.from_tensor_slices(train_images_float_norm).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
type(train_dataset)

#for element in train_dataset:
    #print(element.shape)

# CREATE THE MODEL

# THE GENERATOR
# Conv2DTranspose (also known as Deconvolution)
# https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2DTranspose

# Continue at DECONVOLUTION
# https://iksinc.online/2017/05/06/deconvolution-in-deep-learning/

def make_generator_model():
    model = tf.keras.Sequential()
    model.add(Dense(7*7*256, use_bias=False, input_shape=(100,)))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Reshape((7, 7, 256)))
    assert model.output_shape == (None, 7, 7, 256)  # Note: None is the batch size

    model.add(Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 7, 7, 128)
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 14, 14, 64)
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 28, 28, 1)

    return model


# USE UNTRAINED GENERATOR TO TRAIN AN IMAGE
generator = make_generator_model()
generator.summary()
# generator.load_weights("saved_model/")
noise = tf.random.normal([1, 100]) # TensorShape([1, 100])

generated_image = generator(noise, training=False) # TensorShape([1, 28, 28, 1])

plt.imshow(generated_image[0, :, :, 0], cmap='gray')


# DISCRIMINATOR
def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                     input_shape=[28, 28, 1]))
    model.add(LeakyReLU())
    model.add(Dropout(0.3))

    model.add(Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(LeakyReLU())
    model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(1))

    return model

discriminator = make_discriminator_model()
decision = discriminator(generated_image)
print (decision) # tf.Tensor([[0.00028886]], shape=(1, 1), dtype=float32)
# The model will be trained to output positive values for real images, and negative values for fake images.


# DEFINE THE LOSS AND OPTIMISERS
# This method returns a helper function to compute cross entropy loss
# ref - https://www.tensorflow.org/api_docs/python/tf/keras/losses/BinaryCrossentropy
cross_entropy = BinaryCrossentropy(from_logits=True)
# y_true = [[0., 1.], [0., 0.]]
# y_pred = [[0.6, 0.4], [0.4, 0.6]]
# bce = BinaryCrossentropy()
# bce(y_true, y_pred).numpy()

# DISCRIMINATOR LOSS
def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

# GENERATOR LOSS
def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

# SAVE CHECKPOINTS
checkpoint_dir = './training_checkpoints1'
# checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)
manager = tf.train.CheckpointManager(checkpoint, directory=checkpoint_dir, max_to_keep=2)

# IF WANT TO RELOAD WEIGHTS FROM PREVIOUS TRAINING
reload_from_previous_training = False

if reload_from_previous_training:
    checkpoint.restore(manager.latest_checkpoint) # use this if using manager

# Define the training points
EPOCHS = 4
noise_dim = 100
num_examples_to_generate = 25

# You will reuse this seed overtime (so it's easier)
# to visualize progress in the animated GIF)
seed = tf.random.normal([num_examples_to_generate, noise_dim]) # TensorShape([16, 100])

# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

def train(dataset, epochs):
    print('start training')
    for epoch in range(epochs):
        print('epoch')
        start = time.time()

        for image_batch in dataset:
            print('image_batch')
            train_step(image_batch)

        # Produce images for the GIF as you go
        display.clear_output(wait=True)
        generate_and_save_images(generator,
                                 epoch + 1,
                                 seed)

        # Save the model every 15 epochs
        if (epoch + 1) % 1 == 0:
          # checkpoint.save(file_prefix = checkpoint_prefix)
          manager.save()
          generator.save_weights("saved_model/")

        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

  # Generate after the final epoch
    display.clear_output(wait=True)
    generate_and_save_images(generator,
                           epochs,
                           seed)

# GENERATE AND SAVE IMAGES
def generate_and_save_images(model, epoch, test_input):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training=False)

    plt.figure(figsize=(5, 5))

    for i in range(predictions.shape[0]):
        plt.subplot(5, 5, i+1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()
    plt.close()


#plt.plot([0, 1, 2, 3, 4], [0, 3, 5, 9, 11])
#plt.xlabel('Months')
#plt.ylabel('Books Read')
#plt.savefig('books_read.png')

# # TRAIN THE MODEL
train(train_dataset, EPOCHS)

# # RESTORE THE LATEST CHECKPOINT
    # To restore the models, e.g generator, 
    # 1) create a generator first using make() generator,
    # 2) run all checkpoint codes, then only
    # 3) checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
# checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
# checkpoint.restore(manager.latest_checkpoint) # use this if using manager
