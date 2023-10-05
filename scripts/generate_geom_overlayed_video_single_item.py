from datetime import datetime
import logging
import multiprocessing
import sys

import numpy as np

from scripts.geom_overlay.util import overlay_all_geoms_on_all_video_for_given_time

logging.basicConfig(stream=sys.stdout, level=logging.INFO)


if __name__ == "__main__":
    overlay_all_geoms_on_all_video_for_given_time(
        c="dahlia",
        s=datetime.fromisoformat("2023-09-21T18:32:18.000000+00:00"),
        e=datetime.fromisoformat("2023-09-21T18:32:38.000000+00:00"),
        video_start_end_seconds_offset=3,
    )