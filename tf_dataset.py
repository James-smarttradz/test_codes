# -*- coding: utf-8 -*-
"""
Created on Wed Sep  8 22:43:37 2021

@author: James Ang
"""

import tensorflow.compat.v2 as tf
# import tensorflow_datasets as tfds
from tensorflow.data import Dataset

def data1():
    
    # Construct a tf.data.Dataset
    ds = tfds.load('mnist', split='train', shuffle_files=True) # need to install tfds
    
    # Build your input pipeline
    ds = ds.shuffle(1024).batch(32).prefetch(tf.data.experimental.AUTOTUNE)
    for example in ds.take(1):
        image, label = example["image"], example["label"]
  

def data2():
    
    dataset = Dataset.from_tensor_slices(list(range(20)))
    
    
    # See https://stackoverflow.com/questions/46444018/meaning-of-buffer-size-in-dataset-map-dataset-prefetch-and-dataset-shuffle
    dataset = dataset.shuffle(20)
    
    dataset = dataset.batch(3, drop_remainder= True)
    
    for element in dataset:
        print(element)
      
def main():
    
    data2()
    
if __name__ == '__main__':
    main()