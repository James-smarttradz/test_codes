
Dict = {1: 'Geeks', 2: 'For', 3: 'Geeks'}
print("Dictionary:")
print(Dict)
print(Dict[1],Dict[2],Dict[3])
# If use normal dictionary, there'll be error if use Dict[4]
# will raise a KeyError as the
# 4 is not present in the dictionary
print(Dict[4])


# The functionality of both dictionaries and defualtdict are
# almost same except for the fact that defualtdict never raises a KeyError.
# It provides a default value for the key that does not exists.
from collections import defaultdict

# Function to return a default
# values for keys that is not
# present
def def_value():
    return "Not Present"

d = defaultdict(def_value)
d["a"] = 1
d["b"] = 2
print(d["a"])
print(d["b"])
print(d["c"])

a = {}  # Normal dict
print(a[3])

b = defaultdict(int)
print(b[3]) # equals int() = 0, the default value
b
