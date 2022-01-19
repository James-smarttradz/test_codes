# -*- coding: utf-8 -*-
"""
Created on Sat Jan  1 01:16:31 2022

@author: James Ang
"""

import math
radius = [1,2,3]
area = list(map(lambda x: round(math.pi*(x**2), 2), radius))
print(area)