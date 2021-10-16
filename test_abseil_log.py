# -*- coding: utf-8 -*-

from absl import logging

logging.info('Interesting Stuff')
logging.info('Interesting Stuff with Arguments: %d', 42)

logging.set_verbosity(logging.INFO)
logging.log(logging.DEBUG, 'This will *not* be printed')
logging.set_verbosity(logging.DEBUG)
logging.log(logging.DEBUG, 'This will be printed')

logging.warning('Worrying Stuff')
logging.error('Alarming Stuff')
logging.fatal('AAAAHHHHH!!!!')  # Process exits
