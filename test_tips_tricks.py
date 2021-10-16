# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 11:16:07 2021

@author: James Ang
"""

import numpy as np

""" Test if variable is numpy
"""
x1 = np.array(1)

if isinstance(x1,np.ndarray):
    print('x1: ' + np.array2string(x1))