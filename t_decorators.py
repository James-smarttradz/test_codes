# with decorators, you can add extra features into existing functions
# decorators can change the behaviour of existing function
# ref https://www.youtube.com/watch?v=yNzxXZfkLUA

# EXAMPLE 1
def div(a,b):
    print(a/b)

def smart_div(func):

    def inner(a,b):
        if a<b:
            a,b = b,a
        return func(a,b)
    return inner

div = smart_div(div)

div(1,4)

# EXAMPLE 2 #############################
# Functions are objects that can be passed through as parameters
# https://www.youtube.com/watch?v=r7Dtus7N4pI
def f1():
    print("Called f1")

def f2(f1):
    f1()
    print("Called f2")
f2(f1)

# Wrapper Function
def f1(func):
    def wrapper(*args,**kwargs):
        print("start")
        func(*args,**kwargs)
        print("end")
    # new function is wrapper,
    return wrapper

@f1 # Decorator - same as f1(f)
def f(a):
    print(a)
    print("Hello")

# Function aliasing
# x = f1(f)
f("theere")
# EXAMPLE 3#######################################
import time
def timer(func):
    def wrapper():
        before = time.time()
        func()
        print("FUnction takes",time.time()-before,"seconds")

    return wrapper

@timer
def f1():
    time.sleep(2)

f1()
