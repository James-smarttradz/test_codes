# ref : https://www.geeksforgeeks.org/what-does-the-if-__name__-__main__-do/#:~:text=If%20you%20import%20this%20script,run%20directly%2C%20and%20not%20imported.

a = 1.0
b = 2.0

import argparse

def test():
    # Step 1 CREATE OBJECT
    parser = argparse.ArgumentParser()

    # Step 2 ADD ARGUMENTS
    parser.add_argument("--firstarg", default="hello", help="This is first string", type=str)

    # Step 3 PARGE ARGUMENTS
    args = parser.parse_args()

    # Step 4 GET ARGUMENTS
    print(args.firstarg)


if __name__ == '__main__':
    print(a+b)
    print("run this file directly")
    print(__name__)
    test()
else:
    print(a*b)
    print('function is imported')
    print(__name__)
    test()
