import argparse
import matplotlib.pyplot as plt
parser = argparse.ArgumentParser(description="This is description")

parser.add_argument("-nsamples", help = "Number of samples", type=int)
parser.add_argument("-latent", help = "Latent size", type=int)

# Read arguments from command line
args = parser.parse_args()

a = args.nsamples
print(a)
print(type(a))

print(args.latent)

def cal():
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("-s", "--string", type=str, required=True)
    parser.add_argument("-i", "--integer", type=int, default=50)

    args = parser.parse_args()
    print args.string
    print args.integer
