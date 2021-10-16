# -*- coding: utf-8 -*-
"""
Created on Wed May 26 13:37:03 2021

@author: James Ang
"""

import os
import sys


# tf.__version__

# HOW TO USE?
# python t_tfaddflags.py --sDataSet="mnist"

def tfapp():
    
    # For tensorflow 1.10
    import tensorflow as tf
    cfg = tf.app.flags.FLAGS

    tf.app.flags.DEFINE_string("sDataSet", "cifar10", "cifar10, mnist, flowers,tiny")
    tf.app.flags.DEFINE_float("fBeta2", 0.9, "")

    # cfg(sys.argv)

    print(cfg.sDataSet)
    print(cfg.fBeta2)

##################################
def arg_pars():
    import argparse
    parser = argparse.ArgumentParser(description="This is description")

    parser.add_argument("-nsamples", default=5, help = "Number of samples", type=int)
    parser.add_argument("-latent", default=0.9, help = "Latent size", type=float)

    # Read arguments from command line
    args = parser.parse_args()
    print(args.nsamples)
    print(args.latent)

def sysarg():
    print("This is the name of the program:", sys.argv[0])

    print("Argument List:", str(sys.argv))

    current_path = os.path.dirname(os.path.realpath(sys.argv[0]))

    print(current_path)


def main():
    tfapp()
    # arg_pars()
    # sysarg()

if __name__=='__main__':
    print("runmain")
    main()
