# -*- coding: utf-8 -*-
"""
Created on Sun Aug 22 21:28:03 2021
https://www.tensorflow.org/api_docs/python/tf/function
@author: James Ang

You can use tf.function to make graphs out of your programs.

https://www.machinelearningplus.com/deep-learning/how-use-tf-function-to-speed-up-python-code-tensorflow/
This is why TF2.0 has the tf.function API, to give any user the option to convert a 
regular (eager) python code to a lazy code which is actually speed optimized.


But, why use Graphs?

The primary reason is, graphs allow
your neural network model to be used in environments that dont have a Python interpreter. 
For example, graphs can be deployed in mobile applications or servers. This is not suitable for eagerly executed code.
The second reason is that graphs can speed up computation time. 
They eliminate the need for repetitive initialisation of variables and computation on these variables.

"""
import tensorflow as tf

import traceback
import contextlib
import timeit

# Some helper code to demonstrate the kinds of errors you might encounter.
@contextlib.contextmanager
def assert_raises(error_class):
  try:
    yield
  except error_class as e:
    print('Caught expected exception \n  {}:'.format(error_class))
    traceback.print_exc(limit=2)
  except Exception as e:
    raise e
  else:
    raise Exception('Expected {} to be raised but no error was raised!'.format(
        error_class))


# @tf.function
def f(x, y):
    return x ** 2 + y

def example1():

    x = tf.constant([2, 3])
    y = tf.constant([3, -2])

    return f(x, y)

@tf.function  # The decorator converts `add` into a `Function`.
def add(a, b ):

    return a + b


# v = tf.Variable(1.0)
# with tf.GradientTape() as tape:
#     result = add(v, 3.0)
# tape.gradient(result, v)

@tf.function
def dense_layer(x, w, b):

  return add(tf.matmul(x, w), b)


class SequentialModel(tf.keras.Model):
    def __init__(self, **kwargs):
        super(SequentialModel, self).__init__(**kwargs)
        self.flatten = tf.keras.layers.Flatten(input_shape=(28, 28))
        self.dense_1 = tf.keras.layers.Dense(128, activation="relu")
        self.dropout = tf.keras.layers.Dropout(0.2)
        self.dense_2 = tf.keras.layers.Dense(10)

    def call(self, x):
        x = self.flatten(x)
        x = self.dense_1(x)
        x = self.dropout(x)
        x = self.dense_2(x)
        return x



def main():

    # print(example1())

    # print(add(tf.ones([2, 2]), tf.ones([2, 2])))


    # print(dense_layer(tf.ones([3, 2]), tf.ones([2, 2]), tf.ones([2])))
    
    # Compare time
    input_data = tf.random.uniform([60, 28, 28])

    eager_model = SequentialModel()
    graph_model = tf.function(eager_model)
    
    print("Eager time:", timeit.timeit(lambda: eager_model(input_data), number=10000))  # Eager time: 7.560996899999964
    print("Graph time:", timeit.timeit(lambda: graph_model(input_data), number=10000))  # Graph time: 3.852437400000099

    

if __name__ == "__main__":
    main()
