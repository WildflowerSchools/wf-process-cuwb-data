from datetime import datetime
import logging
import sys

import honeycomb_io
import video_io

from scripts.geom_overlay.util import overlay_all_geoms_on_all_video_for_given_time

logging.basicConfig(stream=sys.stdout, level=logging.INFO)


if __name__ == "__main__":

    OUTPUT_CONCATENATED_VIDEOS_DIR = "../output/concatenated_videos"
    OUTPUT_RAW_VIDEO_SNIPPED_DIR = "../output/raw_videos"

    start = datetime.fromisoformat("2023-10-02T13:40:00-07:00")
    end = datetime.fromisoformat("2023-10-02T13:41:00-07:00")

    environment_id = honeycomb_io.fetch_environment_id(
        environment_name="dahlia",
    )

    video_io.fetch_concatenated_video(
        environment_id=environment_id,
        start=start,
        end=end,
        camera_names=None,
        output_directory=OUTPUT_CONCATENATED_VIDEOS_DIR,
        video_snippet_directory=OUTPUT_RAW_VIDEO_SNIPPED_DIR,
    )

    overlay_all_geoms_on_all_video_for_given_time(
        c="dahlia",
        s=start,
        e=end,
        video_start_end_seconds_offset=0,
    )
