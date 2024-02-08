from datetime import datetime
import logging
import multiprocessing
import sys

import video_io

from scripts.geom_overlay.util import overlay_all_geoms_on_all_video_for_given_time

logging.basicConfig(stream=sys.stdout, level=logging.INFO)


OUTPUT_OVERLAYED_DIR = "../output/overlayed_video"
OUTPUT_FINAL_OVERLAYED_DIR = "../output/final_overlayed_video"
OUTPUT_CONCATENATED_VIDEOS_DIR = "../output/concatenated_videos"
OUTPUT_RAW_VIDEO_SNIPPED_DIR = "../output/raw_videos"

if __name__ == "__main__":
    multiprocessing.freeze_support()

    # events_list = [
    #     [
    #         "dahlia",
    #         "2023-09-07 15:45:04.500000+00:00",
    #         "2023-09-07 15:45:06.600000+00:00",
    #         "wftech-camera-00101",
    #         "wftech-camera-00101",
    #     ],
    #     [
    #         "dahlia",
    #         "2023-09-07 15:45:08.800000+00:00",
    #         "2023-09-07 15:45:32.300000+00:00",
    #         "wftech-camera-00114",
    #         "wftech-camera-00109",
    #     ],
    #     [
    #         "dahlia",
    #         "2023-09-07 15:49:26.300000+00:00",
    #         "2023-09-07 15:50:00.900000+00:00",
    #         "wftech-camera-00103",
    #         "wftech-camera-00103",
    #     ],
    #     [
    #         "dahlia",
    #         "2023-09-07 15:49:49+00:00",
    #         "2023-09-07 15:49:51.200000+00:00",
    #         "wftech-camera-00100",
    #         "wftech-camera-00100",
    #     ],
    #     [
    #         "dahlia",
    #         "2023-09-07 15:50:02.500000+00:00",
    #         "2023-09-07 15:50:06.900000+00:00",
    #         "wftech-camera-00103",
    #         "wftech-camera-00109",
    #     ],
    #     [
    #         "dahlia",
    #         "2023-09-07 15:50:26.900000+00:00",
    #         "2023-09-07 15:50:44.600000+00:00",
    #         "wftech-camera-00103",
    #         "wftech-camera-00101",
    #     ],
    #     [
    #         "dahlia",
    #         "2023-09-07 15:51:52.600000+00:00",
    #         "2023-09-07 15:51:54.600000+00:00",
    #         "wftech-camera-00110",
    #         "wftech-camera-00100",
    #     ],
    #     [
    #         "dahlia",
    #         "2023-09-07 15:58:10.100000+00:00",
    #         "2023-09-07 15:58:14.100000+00:00",
    #         "wftech-camera-00100",
    #         "wftech-camera-00100",
    #     ],
    #     [
    #         "dahlia",
    #         "2023-09-07 16:09:12.200000+00:00",
    #         "2023-09-07 16:09:15.200000+00:00",
    #         "wftech-camera-00111",
    #         "wftech-camera-00110",
    #     ],
    #     [
    #         "dahlia",
    #         "2023-09-07 16:09:22.900000+00:00",
    #         "2023-09-07 16:09:45.100000+00:00",
    #         "wftech-camera-00111",
    #         "wftech-camera-00107",
    #     ],
    #     [
    #         "dahlia",
    #         "2023-09-07 16:14:30.200000+00:00",
    #         "2023-09-07 16:14:35.700000+00:00",
    #         "wftech-camera-00112",
    #         "wftech-camera-00112",
    #     ],
    #     [
    #         "dahlia",
    #         "2023-09-07 16:19:15.100000+00:00",
    #         "2023-09-07 16:19:19+00:00",
    #         "wftech-camera-00100",
    #         "wftech-camera-00100",
    #     ],
    #     [
    #         "dahlia",
    #         "2023-09-07 16:19:41.100000+00:00",
    #         "2023-09-07 16:19:44.500000+00:00",
    #         "wftech-camera-00111",
    #         "wftech-camera-00111",
    #     ],
    #     [
    #         "dahlia",
    #         "2023-09-07 16:32:14.200000+00:00",
    #         "2023-09-07 16:32:17.500000+00:00",
    #         "wftech-camera-00100",
    #         "wftech-camera-00100",
    #     ],
    #     [
    #         "dahlia",
    #         "2023-09-07 16:44:14+00:00",
    #         "2023-09-07 16:44:27.900000+00:00",
    #         "wftech-camera-00102",
    #         "wftech-camera-00112",
    #     ],
    #     [
    #         "dahlia",
    #         "2023-09-07 16:44:39.400000+00:00",
    #         "2023-09-07 16:44:44.200000+00:00",
    #         "wftech-camera-00112",
    #         "wftech-camera-00112",
    #     ],
    #     [
    #         "dahlia",
    #         "2023-09-07 17:08:50.200000+00:00",
    #         "2023-09-07 17:08:56.800000+00:00",
    #         "wftech-camera-00101",
    #         "wftech-camera-00100",
    #     ],
    #     [
    #         "dahlia",
    #         "2023-09-07 17:11:11.100000+00:00",
    #         "2023-09-07 17:11:28.400000+00:00",
    #         "wftech-camera-00100",
    #         "wftech-camera-00102",
    #     ],
    #     [
    #         "dahlia",
    #         "2023-09-07 17:11:33+00:00",
    #         "2023-09-07 17:11:48.100000+00:00",
    #         "wftech-camera-00108",
    #         "wftech-camera-00107",
    #     ],
    #     [
    #         "dahlia",
    #         "2023-09-07 17:12:28.800000+00:00",
    #         "2023-09-07 17:12:34.700000+00:00",
    #         "wftech-camera-00100",
    #         "wftech-camera-00101",
    #     ],
    #     [
    #         "dahlia",
    #         "2023-09-07 17:30:54.100000+00:00",
    #         "2023-09-07 17:31:03.100000+00:00",
    #         "wftech-camera-00100",
    #         "wftech-camera-00100",
    #     ],
    #     [
    #         "dahlia",
    #         "2023-09-07 17:31:08.500000+00:00",
    #         "2023-09-07 17:31:30.700000+00:00",
    #         "wftech-camera-00100",
    #         "wftech-camera-00106",
    #     ],
    #     [
    #         "dahlia",
    #         "2023-09-07 17:31:32.900000+00:00",
    #         "2023-09-07 17:31:36.700000+00:00",
    #         "wftech-camera-00106",
    #         "wftech-camera-00106",
    #     ],
    #     [
    #         "dahlia",
    #         "2023-09-07 17:31:39.500000+00:00",
    #         "2023-09-07 17:31:41.600000+00:00",
    #         "wftech-camera-00107",
    #         "wftech-camera-00107",
    #     ],
    #     [
    #         "dahlia",
    #         "2023-09-07 17:34:20.800000+00:00",
    #         "2023-09-07 17:34:40+00:00",
    #         "wftech-camera-00106",
    #         "wftech-camera-00111",
    #     ],
    #     [
    #         "dahlia",
    #         "2023-09-07 17:37:50.700000+00:00",
    #         "2023-09-07 17:37:52.600000+00:00",
    #         "wftech-camera-00100",
    #         "wftech-camera-00100",
    #     ],
    #     [
    #         "dahlia",
    #         "2023-09-07 17:43:12.300000+00:00",
    #         "2023-09-07 17:43:18.700000+00:00",
    #         "wftech-camera-00100",
    #         "wftech-camera-00100",
    #     ],
    #     [
    #         "dahlia",
    #         "2023-09-07 17:44:09+00:00",
    #         "2023-09-07 17:44:15.100000+00:00",
    #         "wftech-camera-00100",
    #         "wftech-camera-00102",
    #     ],
    #     [
    #         "dahlia",
    #         "2023-09-07 17:47:51.400000+00:00",
    #         "2023-09-07 17:48:05.400000+00:00",
    #         "wftech-camera-00107",
    #         "wftech-camera-00111",
    #     ],
    #     [
    #         "dahlia",
    #         "2023-09-07 17:56:35.300000+00:00",
    #         "2023-09-07 17:56:51+00:00",
    #         "wftech-camera-00100",
    #         "wftech-camera-00102",
    #     ],
    #     [
    #         "dahlia",
    #         "2023-09-07 17:56:39.300000+00:00",
    #         "2023-09-07 17:56:45.400000+00:00",
    #         "wftech-camera-00112",
    #         "wftech-camera-00101",
    #     ],
    #     [
    #         "dahlia",
    #         "2023-09-07 17:56:47.100000+00:00",
    #         "2023-09-07 17:56:48.300000+00:00",
    #         "wftech-camera-00101",
    #         "wftech-camera-00101",
    #     ],
    #     [
    #         "dahlia",
    #         "2023-09-07 17:56:50.700000+00:00",
    #         "2023-09-07 17:56:55.800000+00:00",
    #         "wftech-camera-00100",
    #         "wftech-camera-00100",
    #     ],
    #     [
    #         "dahlia",
    #         "2023-09-07 17:56:59+00:00",
    #         "2023-09-07 17:57:04.900000+00:00",
    #         "wftech-camera-00100",
    #         "wftech-camera-00100",
    #     ],
    #     [
    #         "dahlia",
    #         "2023-09-07 17:57:11.400000+00:00",
    #         "2023-09-07 17:57:14.800000+00:00",
    #         "wftech-camera-00101",
    #         "wftech-camera-00100",
    #     ],
    #     [
    #         "dahlia",
    #         "2023-09-07 17:57:24+00:00",
    #         "2023-09-07 17:57:31+00:00",
    #         "wftech-camera-00100",
    #         "wftech-camera-00100",
    #     ],
    #     [
    #         "dahlia",
    #         "2023-09-07 17:57:35.500000+00:00",
    #         "2023-09-07 17:58:02.800000+00:00",
    #         "wftech-camera-00100",
    #         "wftech-camera-00102",
    #     ],
    #     [
    #         "dahlia",
    #         "2023-09-07 17:58:04.400000+00:00",
    #         "2023-09-07 17:58:35+00:00",
    #         "wftech-camera-00102",
    #         "wftech-camera-00100",
    #     ],
    #     [
    #         "dahlia",
    #         "2023-09-07 17:59:09.200000+00:00",
    #         "2023-09-07 17:59:10.600000+00:00",
    #         "wftech-camera-00102",
    #         "wftech-camera-00102",
    #     ],
    #     [
    #         "dahlia",
    #         "2023-09-07 17:59:18.900000+00:00",
    #         "2023-09-07 17:59:21.500000+00:00",
    #         "wftech-camera-00111",
    #         "wftech-camera-00111",
    #     ],
    #     [
    #         "dahlia",
    #         "2023-09-07 17:59:23.300000+00:00",
    #         "2023-09-07 17:59:26.900000+00:00",
    #         "wftech-camera-00111",
    #         "wftech-camera-00111",
    #     ],
    #     [
    #         "dahlia",
    #         "2023-09-07 18:03:38.300000+00:00",
    #         "2023-09-07 18:03:41.700000+00:00",
    #         "wftech-camera-00100",
    #         "wftech-camera-00100",
    #     ],
    #     [
    #         "dahlia",
    #         "2023-09-07 18:05:29.600000+00:00",
    #         "2023-09-07 18:05:38+00:00",
    #         "wftech-camera-00112",
    #         "wftech-camera-00112",
    #     ],
    #     [
    #         "dahlia",
    #         "2023-09-07 18:05:41.500000+00:00",
    #         "2023-09-07 18:05:49.900000+00:00",
    #         "wftech-camera-00112",
    #         "wftech-camera-00100",
    #     ],
    #     [
    #         "dahlia",
    #         "2023-09-07 18:05:57.100000+00:00",
    #         "2023-09-07 18:06:03.600000+00:00",
    #         "wftech-camera-00100",
    #         "wftech-camera-00100",
    #     ],
    #     [
    #         "dahlia",
    #         "2023-09-07 18:06:05.200000+00:00",
    #         "2023-09-07 18:06:16.800000+00:00",
    #         "wftech-camera-00100",
    #         "wftech-camera-00112",
    #     ],
    #     [
    #         "dahlia",
    #         "2023-09-07 18:07:57.800000+00:00",
    #         "2023-09-07 18:08:04.800000+00:00",
    #         "wftech-camera-00100",
    #         "wftech-camera-00100",
    #     ],
    #     [
    #         "dahlia",
    #         "2023-09-07 18:11:36.100000+00:00",
    #         "2023-09-07 18:11:43.600000+00:00",
    #         "wftech-camera-00101",
    #         "wftech-camera-00101",
    #     ],
    #     [
    #         "dahlia",
    #         "2023-09-07 18:12:30.400000+00:00",
    #         "2023-09-07 18:12:38.700000+00:00",
    #         "wftech-camera-00112",
    #         "wftech-camera-00100",
    #     ],
    #     [
    #         "dahlia",
    #         "2023-09-07 18:12:41+00:00",
    #         "2023-09-07 18:12:42.100000+00:00",
    #         "wftech-camera-00100",
    #         "wftech-camera-00100",
    #     ],
    #     [
    #         "dahlia",
    #         "2023-09-07 18:12:43.800000+00:00",
    #         "2023-09-07 18:12:53.700000+00:00",
    #         "wftech-camera-00100",
    #         "wftech-camera-00101",
    #     ],
    #     [
    #         "dahlia",
    #         "2023-09-07 18:13:59+00:00",
    #         "2023-09-07 18:14:00.500000+00:00",
    #         "wftech-camera-00100",
    #         "wftech-camera-00100",
    #     ],
    #     [
    #         "dahlia",
    #         "2023-09-07 18:15:10.500000+00:00",
    #         "2023-09-07 18:15:12+00:00",
    #         "wftech-camera-00101",
    #         "wftech-camera-00101",
    #     ],
    #     [
    #         "dahlia",
    #         "2023-09-07 18:18:02.200000+00:00",
    #         "2023-09-07 18:18:08+00:00",
    #         "wftech-camera-00112",
    #         "wftech-camera-00110",
    #     ],
    #     [
    #         "dahlia",
    #         "2023-09-07 18:30:43.400000+00:00",
    #         "2023-09-07 18:30:49.400000+00:00",
    #         "wftech-camera-00100",
    #         "wftech-camera-00100",
    #     ],
    #     [
    #         "dahlia",
    #         "2023-09-07 18:31:03.800000+00:00",
    #         "2023-09-07 18:31:13.400000+00:00",
    #         "wftech-camera-00101",
    #         "wftech-camera-00109",
    #     ],
    #     [
    #         "dahlia",
    #         "2023-09-07 18:49:41.300000+00:00",
    #         "2023-09-07 18:49:55+00:00",
    #         "wftech-camera-00105",
    #         "wftech-camera-00110",
    #     ],
    #     [
    #         "dahlia",
    #         "2023-09-07 19:10:03.300000+00:00",
    #         "2023-09-07 19:10:11.100000+00:00",
    #         "wftech-camera-00100",
    #         "wftech-camera-00102",
    #     ],
    #     [
    #         "dahlia",
    #         "2023-09-07 20:06:52.400000+00:00",
    #         "2023-09-07 20:07:07.400000+00:00",
    #         "wftech-camera-00111",
    #         "wftech-camera-00105",
    #     ],
    #     [
    #         "dahlia",
    #         "2023-09-07 20:19:46.900000+00:00",
    #         "2023-09-07 20:19:52+00:00",
    #         "wftech-camera-00105",
    #         "wftech-camera-00105",
    #     ],
    #     [
    #         "dahlia",
    #         "2023-09-07 20:19:56.100000+00:00",
    #         "2023-09-07 20:19:57.100000+00:00",
    #         "wftech-camera-00105",
    #         "wftech-camera-00105",
    #     ],
    #     [
    #         "dahlia",
    #         "2023-09-07 20:20:12.600000+00:00",
    #         "2023-09-07 20:20:22.200000+00:00",
    #         "wftech-camera-00100",
    #         "wftech-camera-00102",
    #     ],
    #     [
    #         "dahlia",
    #         "2023-09-07 20:29:32.100000+00:00",
    #         "2023-09-07 20:29:35.600000+00:00",
    #         "wftech-camera-00111",
    #         "wftech-camera-00111",
    #     ],
    #     [
    #         "dahlia",
    #         "2023-09-07 20:29:37.700000+00:00",
    #         "2023-09-07 20:29:51.100000+00:00",
    #         "wftech-camera-00111",
    #         "wftech-camera-00111",
    #     ],
    #     [
    #         "dahlia",
    #         "2023-09-07 20:32:51.500000+00:00",
    #         "2023-09-07 20:33:24.100000+00:00",
    #         "wftech-camera-00100",
    #         "wftech-camera-00109",
    #     ],
    #     [
    #         "dahlia",
    #         "2023-09-07 20:47:07.800000+00:00",
    #         "2023-09-07 20:47:08.800000+00:00",
    #         "wftech-camera-00100",
    #         "wftech-camera-00100",
    #     ],
    #     [
    #         "dahlia",
    #         "2023-09-07 20:48:13.100000+00:00",
    #         "2023-09-07 20:48:19+00:00",
    #         "wftech-camera-00101",
    #         "wftech-camera-00101",
    #     ],
    #     [
    #         "dahlia",
    #         "2023-09-07 20:48:22.100000+00:00",
    #         "2023-09-07 20:48:28.600000+00:00",
    #         "wftech-camera-00101",
    #         "wftech-camera-00101",
    #     ],
    #     [
    #         "dahlia",
    #         "2023-09-07 20:50:28.800000+00:00",
    #         "2023-09-07 20:50:30.100000+00:00",
    #         "wftech-camera-00100",
    #         "wftech-camera-00100",
    #     ],
    #     [
    #         "dahlia",
    #         "2023-09-07 20:50:31.300000+00:00",
    #         "2023-09-07 20:50:40.900000+00:00",
    #         "wftech-camera-00100",
    #         "wftech-camera-00105",
    #     ],
    #     [
    #         "dahlia",
    #         "2023-09-07 20:52:30.800000+00:00",
    #         "2023-09-07 20:52:32.300000+00:00",
    #         "wftech-camera-00111",
    #         "wftech-camera-00111",
    #     ],
    #     [
    #         "dahlia",
    #         "2023-09-07 20:53:39.600000+00:00",
    #         "2023-09-07 20:53:46.400000+00:00",
    #         "wftech-camera-00100",
    #         "wftech-camera-00112",
    #     ],
    #     [
    #         "dahlia",
    #         "2023-09-07 20:54:01.900000+00:00",
    #         "2023-09-07 20:54:05.600000+00:00",
    #         "wftech-camera-00105",
    #         "wftech-camera-00105",
    #     ],
    #     [
    #         "dahlia",
    #         "2023-09-07 20:57:10+00:00",
    #         "2023-09-07 20:57:13.900000+00:00",
    #         "wftech-camera-00110",
    #         "wftech-camera-00100",
    #     ],
    #     [
    #         "dahlia",
    #         "2023-09-07 21:05:38.900000+00:00",
    #         "2023-09-07 21:05:48.800000+00:00",
    #         "wftech-camera-00102",
    #         "wftech-camera-00111",
    #     ],
    #     [
    #         "dahlia",
    #         "2023-09-07 21:14:01.900000+00:00",
    #         "2023-09-07 21:14:08.200000+00:00",
    #         "wftech-camera-00100",
    #         "wftech-camera-00111",
    #     ],
    #     [
    #         "dahlia",
    #         "2023-09-07 21:21:00.100000+00:00",
    #         "2023-09-07 21:21:09.200000+00:00",
    #         "wftech-camera-00101",
    #         "wftech-camera-00102",
    #     ],
    #     [
    #         "dahlia",
    #         "2023-09-07 21:22:50+00:00",
    #         "2023-09-07 21:24:28.900000+00:00",
    #         "wftech-camera-00101",
    #         "wftech-camera-00106",
    #     ],
    #     [
    #         "dahlia",
    #         "2023-09-07 21:26:22.600000+00:00",
    #         "2023-09-07 21:26:26.500000+00:00",
    #         "wftech-camera-00102",
    #         "wftech-camera-00102",
    #     ],
    #     [
    #         "dahlia",
    #         "2023-09-07 21:27:54.500000+00:00",
    #         "2023-09-07 21:27:55.900000+00:00",
    #         "wftech-camera-00108",
    #         "wftech-camera-00108",
    #     ],
    #     [
    #         "dahlia",
    #         "2023-09-07 21:27:57.800000+00:00",
    #         "2023-09-07 21:28:07.100000+00:00",
    #         "wftech-camera-00108",
    #         "wftech-camera-00109",
    #     ],
    #     [
    #         "dahlia",
    #         "2023-09-07 21:28:09.700000+00:00",
    #         "2023-09-07 21:28:19.500000+00:00",
    #         "wftech-camera-00109",
    #         "wftech-camera-00101",
    #     ],
    #     [
    #         "dahlia",
    #         "2023-09-07 21:34:18.400000+00:00",
    #         "2023-09-07 21:34:19.700000+00:00",
    #         "wftech-camera-00102",
    #         "wftech-camera-00102",
    #     ],
    #     [
    #         "dahlia",
    #         "2023-09-07 21:45:32.800000+00:00",
    #         "2023-09-07 21:45:42.400000+00:00",
    #         "wftech-camera-00110",
    #         "wftech-camera-00111",
    #     ],
    #     [
    #         "dahlia",
    #         "2023-09-07 21:45:52.600000+00:00",
    #         "2023-09-07 21:46:04.100000+00:00",
    #         "wftech-camera-00110",
    #         "wftech-camera-00111",
    #     ],
    #     [
    #         "dahlia",
    #         "2023-09-07 21:47:20+00:00",
    #         "2023-09-07 21:47:40+00:00",
    #         "wftech-camera-00100",
    #         "wftech-camera-00101",
    #     ],
    #     [
    #         "dahlia",
    #         "2023-09-07 21:51:44.500000+00:00",
    #         "2023-09-07 21:52:03.300000+00:00",
    #         "wftech-camera-00105",
    #         "wftech-camera-00111",
    #     ],
    #     [
    #         "dahlia",
    #         "2023-09-07 22:20:41.700000+00:00",
    #         "2023-09-07 22:20:43.900000+00:00",
    #         "wftech-camera-00111",
    #         "wftech-camera-00111",
    #     ],
    #     [
    #         "dahlia",
    #         "2023-09-07 22:23:39+00:00",
    #         "2023-09-07 22:23:50.500000+00:00",
    #         "wftech-camera-00100",
    #         "wftech-camera-00105",
    #     ],
    #     [
    #         "dahlia",
    #         "2023-09-07 22:24:04.500000+00:00",
    #         "2023-09-07 22:24:10.700000+00:00",
    #         "wftech-camera-00100",
    #         "wftech-camera-00112",
    #     ],
    #     [
    #         "dahlia",
    #         "2023-09-07 22:56:31.500000+00:00",
    #         "2023-09-07 22:56:39.900000+00:00",
    #         "wftech-camera-00111",
    #         "wftech-camera-00100",
    #     ],
    #     [
    #         "dahlia",
    #         "2023-09-07 22:56:42.800000+00:00",
    #         "2023-09-07 22:56:50.100000+00:00",
    #         "wftech-camera-00111",
    #         "wftech-camera-00100",
    #     ],
    #     [
    #         "dahlia",
    #         "2023-09-07 22:56:51.500000+00:00",
    #         "2023-09-07 22:56:54.600000+00:00",
    #         "wftech-camera-00100",
    #         "wftech-camera-00100",
    #     ],
    #     [
    #         "dahlia",
    #         "2023-09-07 22:56:57.600000+00:00",
    #         "2023-09-07 22:57:05.400000+00:00",
    #         "wftech-camera-00111",
    #         "wftech-camera-00100",
    #     ],
    #     [
    #         "dahlia",
    #         "2023-09-07 22:57:10.400000+00:00",
    #         "2023-09-07 22:57:13.200000+00:00",
    #         "wftech-camera-00111",
    #         "wftech-camera-00111",
    #     ],
    #     [
    #         "dahlia",
    #         "2023-09-07 22:57:15.200000+00:00",
    #         "2023-09-07 22:57:18.600000+00:00",
    #         "wftech-camera-00111",
    #         "wftech-camera-00111",
    #     ],
    #     [
    #         "dahlia",
    #         "2023-09-07 22:57:21.500000+00:00",
    #         "2023-09-07 22:57:36+00:00",
    #         "wftech-camera-00100",
    #         "wftech-camera-00100",
    #     ],
    #     [
    #         "dahlia",
    #         "2023-09-07 22:57:49.300000+00:00",
    #         "2023-09-07 22:57:52.600000+00:00",
    #         "wftech-camera-00111",
    #         "wftech-camera-00111",
    #     ],
    # ]
    # events_start_and_end_separated = list(np.array(events_list)[:, [0, 1, 3]]) + list(
    #     np.array(events_list)[:, [0, 2, 4]]
    # )
    #
    # events = list(
    #     map(
    #         lambda e: {
    #             "environment": e[0],
    #             "start": datetime.fromisoformat(e[1]),
    #             "end": datetime.fromisoformat(e[1]),
    #             "cameras": e[2],
    #         },
    #         events_start_and_end_separated,
    #     )
    # )

    # teacher_child_events = [
    #     {
    #         "environment": "dahlia",
    #         "start": datetime.fromisoformat("2023-09-20T16:02:00.000+00:00"),
    #         "end": datetime.fromisoformat("2023-09-20T16:03:00.000+00:00"),
    #         "cameras": [
    #             "wftech-camera-00100",
    #             "wftech-camera-00101",
    #             "wftech-camera-00103",
    #             "wftech-camera-00106"
    #         ],
    #     },
    #     {
    #         "environment": "dahlia",
    #         "start": datetime.fromisoformat("2023-09-20T16:31:00.000+00:00"),
    #         "end": datetime.fromisoformat("2023-09-20T16:32:00.000+00:00"),
    #         "cameras": [
    #             "wftech-camera-00100",
    #             "wftech-camera-00101",
    #             "wftech-camera-00103",
    #             "wftech-camera-00106"
    #         ],
    #     },
    #     {
    #         "environment": "dahlia",
    #         "start": datetime.fromisoformat("2023-09-20T16:50:00.000+00:00"),
    #         "end": datetime.fromisoformat("2023-09-20T16:51:00.000+00:00"),
    #         "cameras": [
    #             "wftech-camera-00100",
    #             "wftech-camera-00101",
    #             "wftech-camera-00103",
    #             "wftech-camera-00106"
    #         ],
    #     },
    #     {
    #         "environment": "dahlia",
    #         "start": datetime.fromisoformat("2023-09-20T17:01:00.000+00:00"),
    #         "end": datetime.fromisoformat("2023-09-20T17:02:00.000+00:00"),
    #         "cameras": [
    #             "wftech-camera-00100",
    #             "wftech-camera-00101",
    #             "wftech-camera-00103",
    #             "wftech-camera-00106"
    #         ],
    #     },
    #     {
    #         "environment": "dahlia",
    #         "start": datetime.fromisoformat("2023-09-20T18:50:00.000+00:00"),
    #         "end": datetime.fromisoformat("2023-09-20T18:51:00.000+00:00"),
    #         "cameras": [
    #             "wftech-camera-00100",
    #             "wftech-camera-00101",
    #             "wftech-camera-00103",
    #             "wftech-camera-00106"
    #         ],
    #     },
    #     {
    #         "environment": "dahlia",
    #         "start": datetime.fromisoformat("2023-09-20T18:55:00.000+00:00"),
    #         "end": datetime.fromisoformat("2023-09-20T18:56:00.000+00:00"),
    #         "cameras": [
    #             "wftech-camera-00100",
    #             "wftech-camera-00101",
    #             "wftech-camera-00103",
    #             "wftech-camera-00106"
    #         ],
    #     },
    #     {
    #         "environment": "dahlia",
    #         "start": datetime.fromisoformat("2023-09-20T19:34:00.000+00:00"),
    #         "end": datetime.fromisoformat("2023-09-20T19:35:00.000+00:00"),
    #         "cameras": [
    #             "wftech-camera-00100",
    #             "wftech-camera-00101",
    #             "wftech-camera-00103",
    #             "wftech-camera-00106"
    #         ],
    #     },
    #     {
    #         "environment": "dahlia",
    #         "start": datetime.fromisoformat("2023-09-20T21:55:00.000+00:00"),
    #         "end": datetime.fromisoformat("2023-09-20T21:56:00.000+00:00"),
    #         "cameras": [
    #             "wftech-camera-00100",
    #             "wftech-camera-00101",
    #             "wftech-camera-00103",
    #             "wftech-camera-00106"
    #         ],
    #     },
    #     {
    #         "environment": "dahlia",
    #         "start": datetime.fromisoformat("2023-09-20T22:30:00.000+00:00"),
    #         "end": datetime.fromisoformat("2023-09-20T22:31:00.000+00:00"),
    #         "cameras": [
    #             "wftech-camera-00100",
    #             "wftech-camera-00101",
    #             "wftech-camera-00103",
    #             "wftech-camera-00106"
    #         ],
    #     },
    #     {
    #         "environment": "dahlia",
    #         "start": datetime.fromisoformat("2023-09-20T18:35:00.000+00:00"),
    #         "end": datetime.fromisoformat("2023-09-20T18:36:00.000+00:00"),
    #         "cameras": [
    #             "wftech-camera-00100",
    #             "wftech-camera-00101",
    #             "wftech-camera-00103",
    #             "wftech-camera-00106"
    #         ],
    #     },
    #     {
    #         "environment": "dahlia",
    #         "start": datetime.fromisoformat("2023-09-20T23:37:00.000+00:00"),
    #         "end": datetime.fromisoformat("2023-09-20T23:38:00.000+00:00"),
    #         "cameras": [
    #             "wftech-camera-00100",
    #             "wftech-camera-00101",
    #             "wftech-camera-00103",
    #             "wftech-camera-00106"
    #         ],
    #     },
    # ]

    child_child_event_times = [
        {
            "start": datetime.fromisoformat("2023-10-02 11:19:00-07:00"),
            "end": datetime.fromisoformat("2023-10-02 11:20:00-07:00"),
        },
        {
            "start": datetime.fromisoformat("2023-10-02 10:34:00-07:00"),
            "end": datetime.fromisoformat("2023-10-02 10:35:00-07:00"),
        },
        {
            "start": datetime.fromisoformat("2023-10-02 14:26:00-07:00"),
            "end": datetime.fromisoformat("2023-10-02 14:27:00-07:00"),
        },
        {
            "start": datetime.fromisoformat("2023-10-02 13:11:00-07:00"),
            "end": datetime.fromisoformat("2023-10-02 13:12:00-07:00"),
        },
        {
            "start": datetime.fromisoformat("2023-10-02 10:10:00-07:00"),
            "end": datetime.fromisoformat("2023-10-02 10:11:00-07:00"),
        },
        {
            "start": datetime.fromisoformat("2023-10-02 12:35:00-07:00"),
            "end": datetime.fromisoformat("2023-10-02 12:36:00-07:00"),
        },
        {
            "start": datetime.fromisoformat("2023-10-02 10:11:00-07:00"),
            "end": datetime.fromisoformat("2023-10-02 10:12:00-07:00"),
        },
        {
            "start": datetime.fromisoformat("2023-10-02 08:26:00-07:00"),
            "end": datetime.fromisoformat("2023-10-02 08:27:00-07:00"),
        },
        {
            "start": datetime.fromisoformat("2023-10-02 15:42:00-07:00"),
            "end": datetime.fromisoformat("2023-10-02 15:43:00-07:00"),
        },
        {
            "start": datetime.fromisoformat("2023-10-02 16:11:00-07:00"),
            "end": datetime.fromisoformat("2023-10-02 16:12:00-07:00"),
        },
    ]

    cameras = ["wftech-camera-00100", "wftech-camera-00101", "wftech-camera-00103", "wftech-camera-00106"]

    child_child_events = list(
        map(lambda e: {**e, **{"environment": "dahlia", "cameras": cameras}}, child_child_event_times)
    )

    def _overlay(event):
        output_overlayed_dir = "./output/overlayed_video"

        logging.info(
            f"Generating overlays for, Environment: {event['environment']} Start: {event['start']} End: {event['end']} Cameras: {event['cameras']}"
        )
        df_video_outputs = overlay_all_geoms_on_all_video_for_given_time(
            c=event["environment"],
            s=event["start"],
            e=event["end"],
            cameras=event["cameras"],
            device_ids=event["device_ids"] if "device_ids" in event else None,
            video_start_end_seconds_offset=0,
            display_trays=False,
            display_people=True,
            active_tags_only=True,
        )

        output_path = f"{output_overlayed_dir}/mosaic_{event['environment']}_from-{event['start'].isoformat()}_to-{event['end'].isoformat()}.mp4"
        video_io.combine_videos(video_inputs=df_video_outputs["overlayed_file_path"].to_list(), output_path=output_path)

    with multiprocessing.pool.ThreadPool(processes=2) as pool:
        pool.map(_overlay, child_child_events)

        pool.close()
        pool.join()

    # generate_video_snippets_for_person(
    #     classroom_name="dahlia",
    #     person_device_id="569d24ba-021f-4dcd-a059-f2b899752f07",
    #     start=datetime.strptime("2023-06-30T12:40:18.771697-0700", "%Y-%m-%dT%H:%M:%S.%f%z"),
    #     end=datetime.strptime("2023-06-30T12:43:24.032254-0700", "%Y-%m-%dT%H:%M:%S.%f%z"),
    # )

    # 1) Find all active tag periods
    # 2) Find all the stage-1 passed active pairs
    # 3) Loop across all pairs and draw dots for each pair over the arc of time selected. Cross-reference passed stage-1 pairs in order to paint those a separate color.
