# -*- coding: utf-8 -*-
"""
Created on Sat Jul 31 22:11:00 2021

@author: James Ang
"""

# To disable display “successfully opened CUDA library... ****”
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

# import tensorflow as tf
import tensorflow.compat.v1 as tf
from keras.datasets import mnist
tf.compat.v1.disable_eager_execution()
tf.reset_default_graph()    # When saving graph

# import numpy as np


print(tf.__version__)

def import_mnist(): # IMPORT MNIST DATASET

    (train_images, train_labels), (_, _) = mnist.load_data()
    train_images_float=train_images.reshape(train_images.shape[0],28,28,1).astype('float32')
    
    print(train_images.shape)
    print(train_images.dtype)
    print(train_images_float.shape)
    print(train_images_float.dtype)
    
    return train_images, train_labels

# Training Parameters
lr = 0.001
batch_size = 128
epochs = 100000

# Network Params
image_dim = 784 # 28*28 pixels
gen_hidden_dim = 256
disc_hidden_dim = 256
noise_dim = 100 # Noise data points

def glorot_init(shape):
    return tf.random_normal(shape=shape, stddev= 1 / tf.sqrt(shape[0])/2)

# tf.random.normal([4], 0, 1, tf.float32)
# tf.random.set_seed(5)
# tf.random_normal([4,1], 0, 1, tf.float32, seed=1)

def simple_graph(x1,x2):
    print("hi!")
    a = tf.constant(x1)
    b = tf.constant(x2)
    
    # add
    with tf.name_scope("my_graph_bub"):
        c = tf.add(a,b, name='Add_ab')
        d = tf.add(5,6, name='Add_d')
    
    print(c)
    print(d)

    # mult
    e = tf.multiply(c,d, name="multiply_cd")
    print(e)
    f = tf.divide(e,5)
    
    x = tf.Variable(0,name="x_one")
    y = tf.assign(x, 1)
    
    with tf.Session() as sess: # Don't need if there's eager execution
        

        sess.run(tf.global_variables_initializer()) 
        output = sess.run(f)
        p,q = sess.run([x,y])
        print(p,q)
        # print (sess.run([x,y])) # Just to show can run 2 variables at 1 time.
        # print (sess.run(x))
        print(f"output = {output}")
        
    writer = tf.summary.FileWriter('./name_scope1', sess.graph)
    # writer = tf.summary.FileWriter('./name_scope1')
    # writer.add_graph(sess2.graph)
    writer.close()
        
def interactiv(x1,x2):
    
    sess = tf.InteractiveSession()
    a = tf.constant(x1)
    b = tf.constant(x2)
    
    c = a + b
    
    print("Interactive: c is", c.eval())
    sess.close()
    
    # tf.InteractiveSession._active_session_count

def test_var():
    
    # # Interactive
    # sess = tf.InteractiveSession()
    # a = tf.Variable(initial_value=3)
    # b = tf.constant(2)
    # c = a+b
    
    # # Initialising
    # init = tf.global_variables_initializer()
    # init.run()      # init only works if there's a session
    
    # print("Test Var:", c.eval())
    
    # # change Variable value mid-way
    # # assign will do the initialisation for you
    # final = tf.assign(a,50)
    
    # print(final.eval())
    
    
    # # Test 2
    
    # my_tensor = tf.random_uniform((4,4),0,1)
    # my_var = tf.Variable(initial_value=my_tensor)
    # print(my_var)
    
    # init = tf.global_variables_initializer()
    # init.run()
    # print("my_var:", my_var.eval())
    
    # sess.close()
    
    
    # # Test 3
    
    # x = tf.Variable(0,name="x_one")
    # y = tf.assign(x, 1)
    
    # with tf.Session() as sess1:
        
    #     sess1.run(tf.global_variables_initializer())    # must run within session
    #     print (sess1.run(x))
    #     print (sess1.run(y))
    #     print (sess1.run(x))
    

    # Test 4
    
    a = tf.Variable(0,name="a")
    b = tf.constant(1,name="b")
    mid = tf.add(a,b,name="mid2")
    update_a = tf.assign(a, mid)
    
    
    with tf.Session() as sess2:

        sess2.run(tf.global_variables_initializer())    # must run within session
        print("Initial a:", sess2.run(update_a))
        
        writer = tf.summary.FileWriter('./name_scope1', sess2.graph)
        
        for _ in range(5):
            
            print("Update_a:", sess2.run(update_a))
        #     print("a:", sess2.run(a))
        #     # print("b:", sess2.run(b))
        #     # print("mid (a+b):", sess2.run(mid))
        #     # print("update_a:", sess2.run(update_a))
            
    # writer = tf.summary.FileWriter('./name_scope1', sess2.graph)    # Put here so could see entire graph
    # writer.add_graph(sess2.graph)
    writer.close()

def placehold():
    
    # NOTE: Placeholders are not compatible with eager execution.
    
    x = tf.placeholder(dtype=tf.float32, shape = [None,4], name=None)
    y = x**2
    
    with tf.Session() as sess3:
        x1 = [
            [1,2,3,4],
            [4,5,6,7],
            [7,8,9,10]
            ]
        
        # x2 = np.random.randint([4,3])
        result = sess3.run(y,feed_dict={x :x1})
        print(result)


def main():
    simple_graph(2,5)
    # interactiv(3,9)
    # test_var()
    # placehold()
    


if __name__ == '__main__':
    main()

