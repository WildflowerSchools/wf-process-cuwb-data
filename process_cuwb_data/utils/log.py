import logging
import pandas as pd
import sys


class Logger(object):
    def __init__(self):
        self.set_pandas_output()

        formatter = logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s',
                                      datefmt='%Y-%m-%d %H:%M:%S')
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setLevel(logging.DEBUG)
        stdout_handler.setFormatter(formatter)

        minimal_honeycomb = logging.getLogger('minimal_honeycomb')
        minimal_honeycomb.setLevel(logging.DEBUG)
        minimal_honeycomb.addHandler(stdout_handler)

        minimal_honeycomb = logging.getLogger('honeycomb_io')
        minimal_honeycomb.setLevel(logging.DEBUG)
        minimal_honeycomb.addHandler(stdout_handler)

        gqlpycgen = logging.getLogger('gqlpycgen')  # .client
        gqlpycgen.setLevel(logging.DEBUG)
        gqlpycgen.addHandler(stdout_handler)

        g_logger = logging.getLogger('groundtruth_default_log')
        g_logger.setLevel(logging.DEBUG)
        g_logger.addHandler(stdout_handler)

        self._logger = g_logger

    def set_pandas_output(self, max_rows=100, max_columns=None, width=None, max_colwidth=None):
        pd.set_option('display.max_rows', max_rows)
        pd.set_option('display.max_columns', max_columns)
        pd.set_option('display.width', width)
        pd.set_option('display.max_colwidth', max_colwidth)

    def logger(self):
        return self._logger


logger = Logger().logger()
