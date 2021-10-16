# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 14:49:25 2021

reference:
    (Linear Regression)
    1) https://www.youtube.com/watch?v=PGm8pLp7T40&list=PLdxQ7SoCLQANQ9fQcJ0wnnTzkFsJHlWEj&index=11 
    
    (LR with tensorboard)
    2) https://www.youtube.com/watch?v=JygeABdq2f8&list=PLdxQ7SoCLQANQ9fQcJ0wnnTzkFsJHlWEj&index=12 

@author: James Ang
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
tf.reset_default_graph()

import numpy as np
import matplotlib.pyplot as plt


def lin_reg():
    learning_rate = 0.001
    epochs = 300
    num_samples = 50
    
    x_train = np.linspace(0,20,num_samples)
    y_train = 6*x_train + 7 * np.random.randn(num_samples)
    # plt.scatter(x_train,y_train)
    # plt.plot(x_train,6*x_train)
    
    
    # Create graph Y = w*X + B
    
    X = tf.placeholder(tf.float32, name="x_input")
    Y = tf.placeholder(tf.float32, name="y_input")
    
    W = tf.Variable(initial_value=np.random.randn(),name="weights")
    b = tf.Variable(initial_value=np.random.randn(),name="bias")
    
    # Construct a linear model
    with tf.name_scope("Model") as scope:
        pred = tf.add(tf.multiply(X, W), b)
    
    weight_histogram = tf.summary.histogram("Weights", W)
    bias_histogram = tf.summary.histogram("Biases", b)
    
    with tf.name_scope("Cost") as scope:
        cost = tf.reduce_sum((pred-Y)**2)/(2*num_samples)
    
    cost_summary = tf.summary.scalar("Cost", cost)
    
    with tf.name_scope("Training") as scope:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate, name='GradientDescent').minimize(cost)
    
    # Initialize the Variables
    init = tf.global_variables_initializer()
    
    
    # Merge all the summaries into a single operator
    merged_summaries = tf.summary.merge_all()
    
    with tf.Session() as sess:
        
        sess.run(init)
        
        writer = tf.summary.FileWriter('./lin_reg', sess.graph)
        
        for epoch in range(epochs):
            
            for x,y in zip(x_train,y_train):
                
                sess.run(optimizer, feed_dict={X:x, Y:y})
                
                # Write logs for each epoch
                epoch_summary = sess.run(merged_summaries, feed_dict={X:x, Y:y})
                writer.add_summary(epoch_summary,epoch)
                
            if not epoch%40:    # true if epoch%40 (remainder) is 0 (false), so pick multiple of 40

                print("Epoch:", epoch, "w:",sess.run(W),"b:", sess.run(b), "cost:", 
                      sess.run(cost,feed_dict={X:x_train, Y:y_train}))
        
        print("Optimisation Finished!")
        
        final_cost = sess.run(cost,feed_dict={X:x_train, Y:y_train})
        final_W = sess.run(W)
        final_b = sess.run(b)
        
        print("Final W:" , final_W, "Final bias:", final_b, "Final Cost:", final_cost)
        
        # Plotting
        plt.scatter( x_train, y_train)
        plt.plot( x_train, final_W*x_train + final_b )
        
    
    
    writer.close()
    # cmd: >> tensorboard --logdir lin_reg
    
def main():
    lin_reg()
    
if __name__ == '__main__':
    main()