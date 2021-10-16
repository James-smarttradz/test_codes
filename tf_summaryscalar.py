# -*- coding: utf-8 -*-
"""
Created on Sun Aug  1 13:05:40 2021

@author: James Ang
"""
# To disable display “successfully opened CUDA library ****”
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'


import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
tf.reset_default_graph()

def summary1():
    
    # create the variables
    s_scalar = tf.get_variable(name = "s_scalar", shape = [], 
                               initializer=tf.truncated_normal_initializer(mean=0.0, stddev=1.0))
    
    x_matrix = tf.get_variable('x_matrix', shape=[30, 40], initializer=tf.truncated_normal_initializer(mean=0, stddev=1))
    
    # ____step 1:____ create the summaries
    
    # A scalar summary for the scalar tensor
    first_summary = tf.summary.scalar(name = "First Smary", tensor = s_scalar)
    init = tf.global_variables_initializer()
    
    # A histogram summary for the non-scalar (i.e. 2D or matrix) tensor
    # histogram used to plot the values of a non-scalar tensor.
    # in neural network, used to monitor change in weights and bias
    histogram_summary = tf.summary.histogram('My_histogram_summary', x_matrix)
    
    
    # launch the graph in a session
    with tf.Session() as sess:
        
        # ____step 2:____ creating the writer inside the session
        writer = tf.summary.FileWriter('./name_scope2', sess.graph)
        
        for i in range(100):
            
            # loop over several initializations of the variable
            sess.run(init)
            # ____step 3:____ evaluate the merged summaries
            summary1, summary2 = sess.run([first_summary, histogram_summary])
            # s____step 4:____ add the summary to the writer (i.e. to the event file) to write on the disc
            writer.add_summary(summary1, i)
            # repeat steps 4 for the histogram summary
            writer.add_summary(summary2, i)
            
            
        # sess.run(init)
        # summary = sess.run(first_summary)
        # writer.add_summary(summary,1)
        
        # sess.run(init)
        # summary = sess.run(first_summary)
        # writer.add_summary(summary,2)
        
        # sess.run(init)
        # summary = sess.run(first_summary)
        # writer.add_summary(summary,3)

def summary2():

    # s = tf.get_variable(name = "s_scalar", shape = [], 
    #                             initializer=tf.truncated_normal_initializer(mean=0.0, stddev=1.0))

    # first_summary = tf.summary.scalar(name = "First_Smary", tensor = s)
    init = tf.global_variables_initializer()
    
    with tf.Session() as sess:
    # with test_summary_writer.as_default():
        
        writer = tf.summary.FileWriter('./name_scope2')
        
        sess.run(init)
        # summary = sess.run(first_summary)
        summary = sess.run(tf.summary.scalar('loss', 0.345))
        writer.add_summary(summary,1)
        
        sess.run(init)
        # summary = sess.run(first_summary)
        summary = sess.run(tf.summary.scalar('loss', 0.234))
        writer.add_summary(summary,2)
        
        sess.run(init)
        # summary = sess.run(first_summary)
        summary = sess.run(tf.summary.scalar('loss', 0.123))
        writer.add_summary(summary,3)


def summary3():
    k = tf.placeholder(tf.float32)
    
    # Make a normal distribution, with a shifting mean
    mean_moving_normal = tf.random_normal(shape=[1000], mean=(5*k), stddev=1)
    
    # Record that distribution into a histogram summary
    tf.summary.histogram("normal/moving_mean", mean_moving_normal)
    
    # Setup a session and summary writer
    with tf.Session() as sess:
        writer = tf.summary.FileWriter('./name_scope2')
    
        summaries = tf.summary.merge_all()
    
        # Setup a loop and write the summaries to disk
        N = 3
        for step in range(N):
            
            k_val = step/float(N)
            summ = sess.run(summaries, feed_dict={k: k_val})
            writer.add_summary(summ, global_step=step)
            
        writer.close()

def main():
    # summary1()
    # summary2()
    summary3()
    
if __name__ == '__main__':
    main()