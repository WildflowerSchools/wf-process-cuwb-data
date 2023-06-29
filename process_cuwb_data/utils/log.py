import logging
import os
import sys

import pandas as pd


class Logger:
    def __init__(self):
        self.set_pandas_output()

        formatter = logging.Formatter(fmt="%(asctime)s %(levelname)-8s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setLevel(logging.DEBUG)
        stdout_handler.setFormatter(formatter)

        if os.getenv("ENABLE_HONEYCOMB_LOGS", "false").lower() in ["true", "t"]:
            honeycomb_logger = logging.getLogger("minimal_honeycomb")
            honeycomb_logger.setLevel(logging.DEBUG)
            honeycomb_logger.addHandler(stdout_handler)

            honeycomb_logger = logging.getLogger("honeycomb_io")
            honeycomb_logger.setLevel(logging.DEBUG)
            honeycomb_logger.addHandler(stdout_handler)

            gqlpycgen_logger = logging.getLogger("gqlpycgen")  # .client
            gqlpycgen_logger.setLevel(logging.DEBUG)
            gqlpycgen_logger.addHandler(stdout_handler)

        if os.getenv("ENABLE_GEOM_RENDER_LOGS", "false").lower() in ["true", "t"]:
            geom_render_core_logger = logging.getLogger("geom_render.core")
            geom_render_core_logger.setLevel(logging.DEBUG)
            geom_render_core_logger.addHandler(stdout_handler)

        default_logger = logging.getLogger("process_cuwb_data")
        default_logger.setLevel(logging.DEBUG)
        default_logger.addHandler(stdout_handler)

        self._logger = default_logger

    def set_pandas_output(self, max_rows=100, max_columns=None, width=None, max_colwidth=None):
        pd.set_option("display.max_rows", max_rows)
        pd.set_option("display.max_columns", max_columns)
        pd.set_option("display.width", width)
        pd.set_option("display.max_colwidth", max_colwidth)

    def logger(self):
        return self._logger


logger = Logger().logger()
