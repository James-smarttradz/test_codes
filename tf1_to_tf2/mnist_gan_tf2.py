# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 20:00:46 2021

@author: James Ang
"""


import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

# import tensorflow as tf
import tensorflow as tf
# from keras.datasets import mnist
import input_data
# tf.compat.v1.disable_eager_execution()
tf.compat.v1.reset_default_graph()

import numpy as np
import matplotlib.pyplot as plt


print(tf.__version__)


# def import_mnist(): # IMPORT MNIST DATASET

#     (train_images, train_labels), (_, _) = mnist.load_data()
#     train_images_float=train_images.reshape(train_images.shape[0],28,28,1).astype('float32')

#     print(train_images.shape)
#     print(train_images.dtype)
#     print(train_images_float.shape)
#     print(train_images_float.dtype)

#     return train_images, train_labels


# Training Parameters
lr = 0.001
batch_size = 128
epochs = 100

# Network Params
image_dim = 784 # 28*28 pixels
gen_hidden_dim = 256
disc_hidden_dim = 256
z_noise_dim = 100 # Noise data points

def glorot_init(shape):

    return tf.random.normal(shape=shape, stddev= 1. / tf.sqrt(shape[0]/2.))

# Discriminator
def discriminator(x, weights, bias):

    hidden_layer = tf.nn.relu(tf.add(tf.matmul(x,weights["disc_H"]), bias["disc_H"]))
    final_layer = tf.add(tf.matmul(hidden_layer,weights["disc_final"]), bias["disc_final"])
    disc_output = tf.nn.sigmoid(final_layer)

    return final_layer, disc_output

# Generator
def generator(z, weights, bias):

    hidden_layer = tf.nn.relu(tf.add(tf.matmul(z, weights["gen_H"]), bias["gen_H"]))
    final_layer = tf.add(tf.matmul(hidden_layer, weights["gen_final"]), bias["gen_final"])
    gen_output = tf.nn.sigmoid(final_layer)

    return gen_output


def main():

    # def bias_weights():
    
    bias = {
            "disc_H" : tf.Variable(glorot_init(shape = [disc_hidden_dim])),
            "disc_final" : tf.Variable(glorot_init(shape = [1])),
            "gen_H" : tf.Variable(glorot_init(shape = [gen_hidden_dim])),
            "gen_final" : tf.Variable(glorot_init(shape = [image_dim])),
        }
    
    weights = {
            "disc_H" : tf.Variable(glorot_init(shape = [image_dim, disc_hidden_dim])),
            "disc_final" : tf.Variable(glorot_init(shape = [disc_hidden_dim, 1])),
            "gen_H" : tf.Variable(glorot_init(shape = [z_noise_dim, gen_hidden_dim])),
            "gen_final" : tf.Variable(glorot_init(shape = [gen_hidden_dim, image_dim])),
        }
    
        # return weights, bias
    
    # Call
    # def input_place_holder(z_noise_dim, image_dim):
    
    Z_input = tf.compat.v1.placeholder(dtype=tf.float32, shape = [None, z_noise_dim], name = "noise_z")
    X_input = tf.compat.v1.placeholder(dtype=tf.float32, shape = [None, image_dim], name = "real_input_x")
    
        # return X_input, Z_input
    
    # Call
    # def building_network(X_input, Z_input, weights, bias):
    
    # Building Generator Network
    with tf.compat.v1.name_scope("Generator"):
    
        output_gen = generator(Z_input, weights, bias)
    
    # Building Discriminator Network
    with tf.compat.v1.name_scope("Discriminator"):
    
        real_logit, real_disc = discriminator(X_input, weights, bias)
        fake_logit, fake_disc = discriminator(output_gen, weights, bias)
    
        # return real_logit, real_disc, fake_logit, fake_disc
    
    
    # def loss(real_logit, real_disc, fake_logit, fake_disc):
    
    delta = 0.0001 # to prevent log(0)
    
    with tf.compat.v1.name_scope("Discriminator_loss"):
    
        discriminator_loss = -tf.reduce_mean(input_tensor=tf.math.log(real_disc + delta) + tf.math.log(1. - fake_disc + delta))
    
    with tf.compat.v1.name_scope("Generator_loss"):
    
        generator_loss = -tf.reduce_mean(input_tensor=tf.math.log(fake_disc + delta))
    
        # return discriminator_loss, generator_loss
    
    
    # def summary(discriminator_loss, generator_loss):
    
    # Discriminator - saving data (will call on every epoch later)
    disc_loss_total = tf.compat.v1.summary.scalar(name = "Disc_Total_Loss", tensor = discriminator_loss)
    
    # Generator - saving data (will call on every epoch later)
    gen_loss_total = tf.compat.v1.summary.scalar(name = "Gen_Total_Loss", tensor = generator_loss)
    
        # return disc_loss_total, gen_loss_total
    
    
    # def variables(weights, bias):
    
    disc_var = [
        weights["disc_H"],
        weights["disc_final"],
        bias["disc_H"],
        bias["disc_final"]
        ]
    
    gen_var = [
        weights["gen_H"],
        weights["gen_final"],
        bias["gen_H"],
        bias["gen_final"]
        ]
    
        # return disc_var, gen_var
    
    
    # def optimiser(lr, discriminator_loss, disc_var, generator_loss, gen_var):
    
    with tf.compat.v1.name_scope("Optimiser_Discriminator"):
    
        # https://www.tensorflow.org/api_docs/python/tf/compat/v1/train/AdamOptimizer
        disc_optimize = tf.compat.v1.train.AdamOptimizer(
                                            learning_rate = lr,
                                            name='Adam'
                                            ).minimize(
                                                loss = discriminator_loss,
                                                var_list = disc_var
                                                )
    
    with tf.compat.v1.name_scope("Optimiser_Generator"):
    
        gen_optimize = tf.compat.v1.train.AdamOptimizer(
                                            learning_rate = lr,
                                            name='Adam'
                                            ).minimize(
                                                loss = generator_loss,
                                                var_list = gen_var
                                                )
    
        # return disc_optimize, gen_optimize
    
    
    # def execution(disc_optimize, discriminator_loss, gen_optimize, generator_loss, disc_loss_total, gen_loss_total, X_input, Z_input):
    
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    
    # Initialize the Variables
    init = tf.compat.v1.global_variables_initializer()
    
    with tf.compat.v1.Session() as sess:
    
        sess.run(init)
        # sess.run(disc_optimize)
        # Put writer in session if want to log data to tensorboard
        writer = tf.compat.v1.summary.FileWriter("./log", sess.graph)
    
        for epoch in range(epochs):
            # print(epoch)
    
            # batch size is subset of epoch
            X_batch, _ = mnist.train.next_batch(batch_size)
    
            # Generate noise to feed the discriminator
            Z_noise = sess.run(tf.random.uniform([batch_size, z_noise_dim],minval=-1.,maxval=1.))
    
            # Optimising Discriminator
            _ , disc_loss_epoch = sess.run([disc_optimize, discriminator_loss],
                                          feed_dict = {X_input: X_batch, Z_input: Z_noise })
    
            # Optimising Generator
            _ , gen_loss_epoch = sess.run([ gen_optimize, generator_loss], feed_dict = {Z_input: Z_noise})
    
            # Discriminator Summary
            summary_disc_loss = sess.run(disc_loss_total, feed_dict = {X_input: X_batch, Z_input: Z_noise})
            writer.add_summary(summary_disc_loss,epoch)
    
            # Generator Summary
            summary_gen_loss = sess.run(gen_loss_total, feed_dict={Z_input: Z_noise})
            writer.add_summary(summary_gen_loss,epoch)
    
            if not epoch%20: # same as if epoch%2000 == 0
                print("Epoch:", epoch,
                      "Generator loss", gen_loss_epoch,
                      "Discriminator loss", disc_loss_epoch)
    
        print("Optimisation Completed!")
        writer.close()
        
        # GENERATE DATA from GENERATOR
        
        n = 6
    
        canvas = np.empty((28*n,28*n))
    
        for i in range(n):
            z_noise = np.random.uniform(-1., 1. , [batch_size, z_noise_dim])
    
    
    
            g = sess.run(output_gen, feed_dict={Z_input:z_noise})   
            # output size (128,784), number of images (batch) = 128, image size = 784 (28*28)
    
            # reverse colors
            g = -1 * (g-1)
            
            # Arrange Images to n-by-n canvas
            for j in range(n):
                canvas[i*28 :(i+1) *28, j*28:(j+1)*28] = g[j].reshape([28,28])
    
    
        plt.figure(figsize = (n,n))
        plt.imshow(canvas, origin = "upper", cmap = "gray")
        plt.show()


if __name__ == '__main__':
    main()
