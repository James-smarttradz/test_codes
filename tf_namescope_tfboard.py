# -*- coding: utf-8 -*-
"""
Created on Sun Jul 25 21:32:08 2021

@author: James Ang
"""

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

tf.__version__

with tf.Session() as sess:
    
    with tf.name_scope("Scope_1"):
        a = tf.add(1,2, name ="Scope1_a")
        b = tf.multiply(a,3, name = "Scope1_b")
        
    with tf.name_scope("Scope_2"):
        c = tf.add(4,5, name ="Scope2_c")
        d = tf.multiply(c,6, name = "Scope2_d")
        
    e = tf.add(b,d,"output")
    
with tf.Session() as sess2:
    
    # with tf.name_scope("Scope_1"):
    a = tf.add(1,2, name ="Scope1_a")
    b = tf.multiply(a,3, name = "Scope1_b")
        
    # with tf.name_scope("Scope_2"):
    c = tf.add(4,5, name ="Scope2_c")
    d = tf.multiply(c,6, name = "Scope2_d")
        
    e = tf.add(b,d,"output")

writer = tf.summary.FileWriter('./name_scope')
writer.add_graph(sess.graph)
writer.add_graph(sess2.graph)
writer.close()

# Then go to cmd and type
# conda activate env_tf2
# >> tensorboard --logdir="./name_scope"
# go to chrome - tensorboard:
# http://localhost:6006