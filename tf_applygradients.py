# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 21:37:59 2021
https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Optimizer
@author: James Ang


Processing gradients before applying them
Calling minimize() takes care of both computing the gradients and applying them to the variables. If you want to process the gradients before applying them you can instead use the optimizer in three steps:

1) Compute the gradients with tf.GradientTape.
2) Process the gradients as you wish.
3) Apply the processed gradients with apply_gradients().
    
"""

# Create an optimizer.
opt = tf.keras.optimizers.SGD(learning_rate=0.1)

# Compute the gradients for a list of variables.
with tf.GradientTape() as tape:
  loss = <call_loss_function>
vars = <list_of_variables>
grads = tape.gradient(loss, vars)

# Process the gradients, for example cap them, etc.
# capped_grads = [MyCapper(g) for g in grads]
processed_grads = [process_gradient(g) for g in grads]

# Ask the optimizer to apply the processed gradients.
opt.apply_gradients(zip(processed_grads, var_list))