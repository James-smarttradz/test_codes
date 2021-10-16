import argparse
# python test_argparse3.py 1
parser = argparse.ArgumentParser(description="This is description")

parser.add_argument("num", help="This is the number you want to input", type=int)

args = parser.parse_args()

print("The output is:",args.num)
