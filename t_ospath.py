import os
import sys

# sys.path.append(os.path.dirname(os.path.abspath(__file__)))
print("File")
print(__file__)

print("abs.path")
print(os.path.abspath(__file__))

print("dirname")
print(os.path.dirname(os.path.abspath(__file__)))

print("syspath")
print(sys.path)
