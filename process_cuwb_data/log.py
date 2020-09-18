import logging
import sys


class Logger(object):
    def __init__(self):
        formatter = logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s',
                                      datefmt='%Y-%m-%d %H:%M:%S')
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setLevel(logging.DEBUG)
        stdout_handler.setFormatter(formatter)
        g_logger = logging.getLogger('groundtruth_default_log')
        g_logger.setLevel(logging.DEBUG)
        g_logger.addHandler(stdout_handler)
        self._logger = g_logger

    def logger(self):
        return self._logger


logger = Logger().logger()
