# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 14:49:25 2021

reference:
    (Linear Regression)
    1) https://www.youtube.com/watch?v=7p-NjKqmWj8&list=PLC3dwsznxb3IzbjgoNdnurf5SixQakdAz&index=9

@author: James Ang
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] ='2'

import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt


class regression():
    
    def __init__(self):
        
        self.a = tf.Variable(initial_value=0, dtype=tf.float32)
        
        self.b = tf.Variable(initial_value=0, dtype=tf.float32)
        
    def __call__(self, X): # Directly call
        
        # Equation
        # yhat = b*X + a
        
        self.X = tf.convert_to_tensor(X, dtype=tf.float32)
        self.yhat = tf.add(tf.multiply(self.b, self.X), self.a)
        
        return self.yhat
        
def loss_func(y_true, y_hat):
    
    # sum of squares error
    sse = tf.reduce_sum(tf.square(tf.subtract(y_true, y_hat)))
    
    return sse

def train(model, x_train, y_true, learning_rate = 0.001):    
    
    # GradientTape
    with tf.GradientTape() as g:
        
        # g.watch(inputs)
        y_hat = model(x_train)
        sse = loss_func(y_true, y_hat)
        
        da,db = g.gradient(sse, [model.a, model.b])
        
    # update a and b values
    model.a.assign_sub(da*learning_rate)
    model.b.assign_sub(db*learning_rate)
    
    return sse


def plotting(model,x,y):
    
    plt.scatter(x,y)
    plt.plot(x, model(x))

def main():
    
    lr = 0.00001
    epochs = 500
    num_samples = 50
    
    x_train = np.linspace(0,20,num_samples)
    y_true_np = 6*x_train + 7 * np.random.randn(num_samples)
    y_true = tf.convert_to_tensor(y_true_np, dtype=tf.float32)
    # plt.scatter(x_train,y_train)
    # plt.plot(x_train,6*x_train)
    
    model = regression()
    
    for epoch in range(epochs):
        
        sse = train(model, x_train, y_true, learning_rate = lr)
        
        
        if not epoch%5:
            plt.scatter(x_train, y_true)
            plt.plot(x_train, model(x_train))
            print("Epoch: %d, a: %0.3f, b: %0.3f, Cost: %0.1f" %(epoch, model.a, model.b, sse))

if __name__ == '__main__':
    main()