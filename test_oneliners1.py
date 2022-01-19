# -*- coding: utf-8 -*-
"""
Created on Sat Oct 23 01:15:28 2021

https://allwin-raju-12.medium.com/50-python-one-liners-everyone-should-know-182ea7c8de9d

@author: James Ang
"""

#%% Anagram
from collections import Counter

s1 = 'below'
s2 = 'elbow'

print('anagram') if Counter(s1) == Counter(s2) else print('not an anagram')

print('anagram') if sorted(s1) == sorted(s2) else print('not an anagram')

#%% Binary to decimal

decimal = int('1010', 2)
print(decimal) #10

#%% Converting string to lower case

"Hi my name is Allwin".lower()
# 'hi my name is allwin'
"Hi my name is Allwin".casefold()
# 'hi my name is allwin'

#%% Converting string to upper case
"hi my name is Allwin".upper()
# 'HI MY NAME IS ALLWIN'

#%% Converting string to byte

a="convert string to bytes using encode method".encode()
# b'convert string to bytes using encode method'

#%% Copy files

import shutil; shutil.copyfile('source.txt', 'dest.txt')

#%% Quicksort

qsort = lambda l : l if len(l)<=1 else qsort([x for x in l[1:] if x < l[0]]) + [l[0]] + qsort([x for x in l[1:] if x >= l[0]])
