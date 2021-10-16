import tensorflow as tf

tf.__version__

@tf.function
def add(a,b):
    return(a+b)

add(tf.ones([2,2]),tf.ones([2,2]))



