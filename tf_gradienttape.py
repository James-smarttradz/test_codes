# -*- coding: utf-8 -*-
"""
Created on Sun Aug 22 22:45:49 2021

https://www.tensorflow.org/guide/autodiff

@author: James Ang
"""

import tensorflow as tf
import matplotlib.pyplot as plt

def example1():
    
    x = tf.constant(3.0)
    
    with tf.GradientTape() as g:
        
        g.watch(x)
        y = x * x + 5*x*x*x

    # Once you've recorded some operations, use GradientTape.gradient(target, sources) 
    # to calculate the gradient of some target (often a loss) relative to some 
    # source (often the model's variables):
    dy_dx = g.gradient(y, x) # gradisnt of the function y at x = 3
    
    print(dy_dx)
    
    
def example2():
    
    x = tf.constant(5.0)
    
    with tf.GradientTape() as g:
        
        g.watch(x)
        
        with tf.GradientTape() as gg:
            
            gg.watch(x)
            y = x * x
            
        dy_dx = gg.gradient(y, x)  # dy_dx = 2 * x
        
    d2y_dx2 = g.gradient(dy_dx, x)  # d2y_dx2 = 2
    
    print(dy_dx)
    print(d2y_dx2)
    

def example3():
    
    # tf.GradientTape works as easily on any tensor
    
    w = tf.Variable(tf.random.normal((3, 2)), name='w')
    b = tf.Variable(tf.zeros(2, dtype=tf.float32), name='b')
    x = [[1., 2., 3.]]
    
    with tf.GradientTape(persistent=True) as tape:
        
        y = x @ w + b
        loss = tf.reduce_mean(y**2)
      
    [dl_dw, dl_db] = tape.gradient(loss, [w, b])
    
    print([dl_dw, dl_db])
    
    print(w.shape)
    print(dl_dw.shape)


def example4():
    
    # Controlling what the tape watches
    # A trainable variable
    x0 = tf.Variable(3.0, name='x0')
    # Not trainable
    x1 = tf.Variable(3.0, name='x1', trainable=False)
    # Not a Variable: A variable + tensor returns a tensor.
    x2 = tf.Variable(2.0, name='x2') + 1.0
    # Not a variable
    x3 = tf.constant(3.0, name='x3')
    
    with tf.GradientTape() as tape:
        tape.watch([x1,x2]) # if still want the gradient from tf.constant or tf.tensor
        y = (x0**2) + (x1**2) + (x2**3)
    
    grad = tape.gradient(y, [x0, x1, x2, x3])
    
    for g in grad:
        print(g)
      
    print([var.name for var in tape.watched_variables()])

def example5():
    
    x = tf.linspace(-10.0, 10.0, 200+1)

    with tf.GradientTape() as tape:
        tape.watch(x)
        y = tf.nn.sigmoid(x)
    
    dy_dx = tape.gradient(y, x)
    
    plt.plot(x, y, label='y')
    plt.plot(x, dy_dx, label='dy/dx')
    plt.legend()
    _ = plt.xlabel('x')
    
def example6():
    
    # single gradient - persistent
    x = tf.constant(3.0)
    
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(x)
        y = x**2
        z = y**2
    dydx = tape.gradient(y,x)
    dzdx = tape.gradient(z,x)
        
    print(dydx)
    print(dzdx)


def example7():
    
    # higher order gradient
    x = tf.constant(4.0)
    
    with tf.GradientTape() as g:
        g.watch(x)

        
        with tf.GradientTape() as gg:
            gg.watch(x)
            y = x**3
        
            dydx = gg.gradient(y,x)
        
        d2y_dx2 = g.gradient(dydx,x)
        
    print(dydx)
    print(d2y_dx2)
    
def example8():
    
    # single gradient - persistent
    x = tf.constant(3.0)
    
    with tf.GradientTape() as tape:
        tape.watch(x)
        y = x**2

        dydx = tape.gradient(y,x)

    print(dydx)


def main():
    # example1()
    
    # example2()
    
    # example3()
    
    # example4()
    
    # example5()
    
    example6()
    
if __name__ == '__main__':
    main()