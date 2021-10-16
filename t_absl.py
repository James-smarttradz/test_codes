# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 13:56:14 2021
ref: https://abseil.io/docs/python/guides/flags
https://abseil.io/docs/cpp/guides/flags
@author: James Ang

to use:
python t_absl.py -name James
python t_absl.py --help
"""

from absl import app
from absl import flags

FLAGS = flags.FLAGS

# ABSL_FLAG(type, name, default, help-text)


# Flag names are globally defined!  So in general, we need to be
# careful to pick names that are unlikely to be used by other libraries.
# If there is a conflict, we'll get an error at import time.
flags.DEFINE_string('name', 'Jane Random', 'Your name.')
flags.DEFINE_integer('age', None, 'Your age in years.', lower_bound=0, upper_bound=10) # CAN USE LOWER BOUND
# If out of bound will return..
# FATAL Flags parsing error: flag --age=-5: -5 is not a non-negative integer

flags.DEFINE_boolean('debug', False, 'Produces debugging output.')

flags.DEFINE_enum('job', 'running', ['running', 'stopped'], 'Job status.')
# IF job not in list, e.g job=engineer, will return error..
# FATAL Flags parsing error: flag --job=engineer: value should be one of <running|stopped>


def main(argv):

    # FLAGS MUST BE PUT INSIDE HERE!! OR ELSE WILL GET ERROR SUCH AS BELOW
    # UnparsedFlagAccessError: Trying to access flag --epoch1 before flags were parsed.

    # USAGE
    # python t_absl.py -nodebug
    # python t_absl.py -debug

    if FLAGS.debug:     # IF TRUE, RUN THIS
        print('non-flag arguments:', argv)
    else:
        print("Debug is False")



    print('Happy Birthday', FLAGS.name)

    if FLAGS.age is not None:
        print('You are %d years old, and your job is %s' % (FLAGS.age, FLAGS.job))


if __name__ == '__main__':
  app.run(main)
