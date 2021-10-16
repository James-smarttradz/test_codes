"""
ref: https://www.youtube.com/watch?v=yTJ8QydIgVQ&t=23s
"""

import tensorflow as tf

# gfile = tf.gfile            # TF 1.0
gfile = tf.io.gfile.GFile   # TF 2.0

dirpath = "testgfile_dir"
filepath = "testgfile_dir/file"

tf.io.gfile.mkdir(dirpath)

with tf.io.gfile.GFile(filepath, "w") as f:
    f.write("Welcome to TF Training\n")

with tf.io.gfile.GFile(filepath) as f:
    print(f.read())

print("Entries:")
for entry in tf.io.gfile.walk(dirpath):
    print("\t{}".format(entry))
