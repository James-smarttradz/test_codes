# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 18:50:18 2022

@author: James Ang
"""


# Building new exception using built-in Exception class
class JamesError(Exception):
    pass



x = 'hello'

try:
  print("x value: {}".format(x))
  
  if not type(x) is int:
      raise JamesError("Only integers are allowed")
  
except TypeError:
  print("Type error occurs")
except JamesError:
  print("James error occurred. So beware James will come and find you!")
except:
  print("An exception occurred")
  


# x=1

try:
  print(x)
except NameError:
  print("Variable x is not defined")
except:
  print("Something else went wrong")
else:
  print("Nothing went wrong")
finally:
  print("The 'try except' is finished")
  
  
x = -1

if x < 0:
  raise Exception("Sorry, no numbers below zero")
  
x = "hello"

if not type(x) is int:
  raise TypeError("Only integers are allowed")