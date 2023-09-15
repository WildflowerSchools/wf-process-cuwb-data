import functools
import tempfile
from datetime import datetime, timedelta
from functools import partial
import logging
import multiprocessing
import os
import sys
from pathlib import Path

import cv_utils
import ffmpeg
import honeycomb_io
import pandas as pd
import numpy as np

from process_cuwb_data import (
    CameraUWBLineOfSight,
    HoneycombCachingClient,
)
from process_cuwb_data.geom_render import fetch_geoms_2d
import video_io

from geom_render import GeomCollection2D

logging.basicConfig(stream=sys.stdout, level=logging.INFO)


OUTPUT_OVERLAYED_DIR = "../output/overlayed_video"
OUTPUT_FINAL_OVERLAYED_DIR = "../output/final_overlayed_video"
OUTPUT_CONCATENATED_VIDEOS_DIR = "../output/concatenated_videos"
OUTPUT_RAW_VIDEO_SNIPPED_DIR = "../output/raw_videos"


def overlay_geoms_on_video(df_video_snippets_chunk, geoms, material_event_start, material_event_end):
    df_video_snippets_chunk.reset_index(inplace=True, drop=True)
    for idx, video in df_video_snippets_chunk.iterrows():
        video_input = cv_utils.VideoInput(input_path=video["file_path"], start_time=material_event_start)

        video_frame_count = video_input.video_parameters.frame_count
        geom_count = geoms[video["camera_device_id"]]["geom"].coordinates.shape[0]

        logging.info(
            f"Overlaying geoms on video for {geoms[video['camera_device_id']]['device_name']} from {material_event_start} to {material_event_end}. File: '{video['file_path']}'. Frames: {video_frame_count}"
        )
        if video_frame_count > geom_count:
            logging.warning(
                f"Video frame count ('{video_frame_count}') doesn't match geom count ('{geom_count}') for {geoms[video['camera_device_id']]['device_name']} from {material_event_start} to {material_event_end}. File: '{video['file_path']}'."
            )

        os.makedirs(os.path.dirname(df_video_snippets_chunk.iloc[idx]["overlayed_file_path"]), exist_ok=True)

        video_input.close()

        geom_collection_2d = geoms[video["camera_device_id"]]["geom"]
        geom_collection_2d.overlay_video(
            input_path=video["file_path"],
            output_path=df_video_snippets_chunk.iloc[idx]["overlayed_file_path"],
            start_time=material_event_start,
            progress_bar=True,
        )


def overlay_all_geoms_on_all_video_for_given_time(c, s, e, cameras=None, video_start_end_seconds_offset=0):
    environment_id = honeycomb_io.fetch_environment_id(
        environment_name=c,
    )

    df_cameras = honeycomb_io.fetch_camera_info(environment_id=environment_id, start=s, end=e)

    geoms = fetch_geoms_2d(
        environment_name=c,
        start_time=s - timedelta(seconds=video_start_end_seconds_offset),
        end_time=e + timedelta(seconds=video_start_end_seconds_offset),
        smooth=True,
        device_ids=None,
    )
    if cameras:
        geoms = dict(filter(lambda e: e[1]["device_name"] in cameras, geoms.items()))

    df_video_snippets = video_io.fetch_concatenated_video(
        environment_id=environment_id,
        start=s - timedelta(seconds=video_start_end_seconds_offset),
        end=e + timedelta(seconds=video_start_end_seconds_offset),
        camera_names=cameras,
        output_directory=OUTPUT_CONCATENATED_VIDEOS_DIR,
        video_snippet_directory=OUTPUT_RAW_VIDEO_SNIPPED_DIR,
    )
    if df_video_snippets is None:
        logging.error(f"Missing video for {c} - from {s} - {e}")
        return

    df_video_outputs = df_video_snippets.copy()

    def generate_output_file_names(x):
        return f"{OUTPUT_OVERLAYED_DIR}/{c}_{s.strftime('%m%d%YT%H%M%S')}_{e.strftime('%m%d%YT%H%M%S')}_{x['camera_device_id']}.mp4"

    df_video_outputs["overlayed_file_path"] = df_video_outputs.apply(generate_output_file_names, axis=1)

    cpu_count = multiprocessing.cpu_count()

    processes_count = cpu_count if df_video_outputs.shape[0] > cpu_count else df_video_outputs.shape[0]
    pool = multiprocessing.Pool(processes=processes_count)
    chunk_size = int(df_video_outputs.shape[0] / processes_count)

    overlay_geoms_partial = partial(
        overlay_geoms_on_video,
        geoms=geoms,
        material_event_start=s - timedelta(seconds=video_start_end_seconds_offset),
        material_event_end=e + timedelta(seconds=video_start_end_seconds_offset),
    )

    chunks = [
        df_video_outputs.iloc[df_video_outputs.index[ii : ii + chunk_size]]
        for ii in range(0, df_video_outputs.shape[0], chunk_size)
    ]
    pool.map(overlay_geoms_partial, chunks)
    pool.close()
    pool.join()


def generate_video_snippets_for_person(
    classroom_name: str,
    person_device_id: str,
    start: datetime,
    end: datetime,
):
    honeycomb_caching_client = HoneycombCachingClient()
    camera_info = honeycomb_caching_client.fetch_camera_devices(environment_name=classroom_name, start=start, end=end)
    default_camera_device_id = camera_info.index.unique().tolist()[0]

    # Fetch "best camera" video over arc of day
    df_best_camera = pd.DataFrame(columns=["time", "best_camera"])

    best_camera_partial = functools.partial(
        CameraUWBLineOfSight, environment_name=classroom_name, z_axis_override=0.5, position_window_seconds=6
    )
    seconds_frequency = 5

    time_range = pd.date_range(start=start, end=end, freq=f"{seconds_frequency}S")
    if end in time_range:
        time_range = time_range[:-1]

    last_best_camera_id = default_camera_device_id
    for time in time_range:
        try:
            uwb_line_of_sight = best_camera_partial(
                timestamp=time.to_pydatetime(),
                default_camera_device_id=last_best_camera_id,
                tag_device_id=person_device_id,
            )
            best_camera_device_id = uwb_line_of_sight.best_camera_view().index[0]
            df_best_camera = df_best_camera.append(
                {"time": time, "best_camera": best_camera_device_id, "uwb_line_of_sight": uwb_line_of_sight},
                ignore_index=True,
            )
            last_best_camera_id = best_camera_device_id
        except ValueError:
            df_best_camera = df_best_camera.append(
                {"time": time, "best_camera": last_best_camera_id, "uwb_line_of_sight": None}, ignore_index=True
            )

    df_all_video_snippets = pd.DataFrame()
    # Fetch and concat "best camera" video into single video
    for index, row in df_best_camera.iterrows():
        logging.info(f"Generating concatenated video for {classroom_name}")
        df_video_snippets = video_io.fetch_concatenated_video(
            environment_name=classroom_name,
            start=row["time"].to_pydatetime(),
            end=row["time"].to_pydatetime() + timedelta(seconds=seconds_frequency),
            camera_names=camera_info.loc[row["best_camera"]]["device_name"],
            output_directory=OUTPUT_CONCATENATED_VIDEOS_DIR,
            video_snippet_directory=OUTPUT_RAW_VIDEO_SNIPPED_DIR,
        )

        if df_video_snippets is None:
            logging.error(
                f"Missing video for {classroom_name} - from {row['time']} - {row['time'] + timedelta(seconds=seconds_frequency)}"
            )
            continue

        df_video_snippets["start"] = row["time"].to_pydatetime()
        df_video_snippets["end"] = row["time"].to_pydatetime() + timedelta(seconds=seconds_frequency)
        df_video_snippets["best_camera_id"] = row["best_camera"]
        df_all_video_snippets = df_all_video_snippets.append(df_video_snippets)

    # Fetch geoms
    geom_collection_3d, geoms = fetch_geoms_2d(
        environment_name=classroom_name,
        start_time=start,
        end_time=end,
        smooth=True,
        device_ids=[person_device_id],
    )

    # Build a new geom object that samples snippets of time from geoms for each "best" camera
    coordinate_for_concatenated_video = np.empty([0, 1, 2])
    last_index = 0
    for idx, row in df_all_video_snippets.iterrows():
        geom_coordinates = geoms[row["best_camera_id"]]["geom"].coordinates
        frame_count = int((row["end"] - row["start"]).total_seconds() * 10)

        new_index = last_index + frame_count
        new_coordinates = geom_coordinates[last_index:new_index, :]
        coordinate_for_concatenated_video = np.append(coordinate_for_concatenated_video, new_coordinates, axis=0)

        print(
            f"Preparing Geom for {classroom_name} Last index {last_index} Frame count: {frame_count} Range: [{last_index}:{(last_index + (frame_count-1))}] New coords: {len(new_coordinates)} Total Coords: {len(coordinate_for_concatenated_video)} Person ID: {person_device_id} Camera ID: {row['best_camera_id']} Start: {row['start']} End: {row['end']}"
        )

        last_index += frame_count

    time_index = pd.date_range(start=start, end=end, freq=f"{100}L")
    if end in time_index:
        time_index = time_index[:-1]
    print(
        f"Length time_index: {len(time_index)} Length coordinates: {len(coordinate_for_concatenated_video)} Start: {start} End: {end}"
    )
    geoms_for_concatenated_video = GeomCollection2D(
        time_index=time_index,
        geom_list=geoms[next(iter(geoms))]["geom"].geom_list,
        coordinates=coordinate_for_concatenated_video,
    )

    final_video_files = []
    for idx, video_snippet_row in df_all_video_snippets.iterrows():
        file_path_obj = Path(video_snippet_row["file_path"])
        output_path = f"{OUTPUT_OVERLAYED_DIR}/{file_path_obj.stem}_overlayed{file_path_obj.suffix}"

        # final_video_files.append(output_path)
        final_video_files.append(video_snippet_row["file_path"])

        geom = geoms[video_snippet_row["camera_device_id"]]["geom"]

        geom.overlay_video(
            input_path=video_snippet_row["file_path"],
            output_path=output_path,
            start_time=video_snippet_row["start"],
            include_timestamp=True,
            progress_bar=True,
            notebook=False,
        )

    logging.info("Generating final concatenated video file WITHOUT overlayed snippets")
    video_concatenated_output_path = f"{OUTPUT_CONCATENATED_VIDEOS_DIR}/{classroom_name}_{person_device_id}_{start.strftime('%m%d%YT%H%M%S%z')}_{end.strftime('%m%d%YT%H%M%S%z')}.mp4"

    with tempfile.NamedTemporaryFile() as tmp_concat_demuxer_file:
        for file_path in final_video_files:
            tmp_concat_demuxer_file.write(str.encode(f"file 'file:{file_path}'\n"))
            tmp_concat_demuxer_file.flush()

        ffmpeg.input(f"file:{tmp_concat_demuxer_file.name}", format="concat", safe=0, r=10,).output(
            f"file:{video_concatenated_output_path}",
            #             vcodec="h264_videotoolbox",
            #             video_bitrate="3m",
            r=10,
            video_track_timescale=10,
        ).overwrite_output().global_args("-hide_banner", "-loglevel", "warning").run()

    logging.info("Generating final video file WITH overlayed snippets")

    final_overlayed_video_output_path = f"{OUTPUT_FINAL_OVERLAYED_DIR}/{classroom_name}_{person_device_id}_{start.strftime('%m%d%YT%H%M%S%z')}_{end.strftime('%m%d%YT%H%M%S%z')}.mp4"
    geoms_for_concatenated_video.overlay_video(
        input_path=video_concatenated_output_path,
        output_path=final_overlayed_video_output_path,
        start_time=start,
        include_timestamp=True,
        progress_bar=True,
        notebook=False,
    )


if __name__ == "__main__":
    multiprocessing.freeze_support()

    events_list = [
        ["dahlia","2023-09-07 15:45:04.500000+00:00","2023-09-07 15:45:06.600000+00:00","wftech-camera-00101","wftech-camera-00101"],
        ["dahlia","2023-09-07 15:45:08.800000+00:00","2023-09-07 15:45:32.300000+00:00","wftech-camera-00114","wftech-camera-00109"],
        ["dahlia","2023-09-07 15:49:26.300000+00:00","2023-09-07 15:50:00.900000+00:00","wftech-camera-00103","wftech-camera-00103"],
        ["dahlia","2023-09-07 15:49:49+00:00","2023-09-07 15:49:51.200000+00:00","wftech-camera-00100","wftech-camera-00100"],
        ["dahlia","2023-09-07 15:50:02.500000+00:00","2023-09-07 15:50:06.900000+00:00","wftech-camera-00103","wftech-camera-00109"],
        ["dahlia","2023-09-07 15:50:26.900000+00:00","2023-09-07 15:50:44.600000+00:00","wftech-camera-00103","wftech-camera-00101"],
        ["dahlia","2023-09-07 15:51:52.600000+00:00","2023-09-07 15:51:54.600000+00:00","wftech-camera-00110","wftech-camera-00100"],
        ["dahlia","2023-09-07 15:58:10.100000+00:00","2023-09-07 15:58:14.100000+00:00","wftech-camera-00100","wftech-camera-00100"],
        ["dahlia","2023-09-07 16:09:12.200000+00:00","2023-09-07 16:09:15.200000+00:00","wftech-camera-00111","wftech-camera-00110"],
        ["dahlia","2023-09-07 16:09:22.900000+00:00","2023-09-07 16:09:45.100000+00:00","wftech-camera-00111","wftech-camera-00107"],
        ["dahlia","2023-09-07 16:14:30.200000+00:00","2023-09-07 16:14:35.700000+00:00","wftech-camera-00112","wftech-camera-00112"],
        ["dahlia","2023-09-07 16:19:15.100000+00:00","2023-09-07 16:19:19+00:00","wftech-camera-00100","wftech-camera-00100"],
        ["dahlia","2023-09-07 16:19:41.100000+00:00","2023-09-07 16:19:44.500000+00:00","wftech-camera-00111","wftech-camera-00111"],
        ["dahlia","2023-09-07 16:32:14.200000+00:00","2023-09-07 16:32:17.500000+00:00","wftech-camera-00100","wftech-camera-00100"],
        ["dahlia","2023-09-07 16:44:14+00:00","2023-09-07 16:44:27.900000+00:00","wftech-camera-00102","wftech-camera-00112"],
        ["dahlia","2023-09-07 16:44:39.400000+00:00","2023-09-07 16:44:44.200000+00:00","wftech-camera-00112","wftech-camera-00112"],
        ["dahlia","2023-09-07 17:08:50.200000+00:00","2023-09-07 17:08:56.800000+00:00","wftech-camera-00101","wftech-camera-00100"],
        ["dahlia","2023-09-07 17:11:11.100000+00:00","2023-09-07 17:11:28.400000+00:00","wftech-camera-00100","wftech-camera-00102"],
        ["dahlia","2023-09-07 17:11:33+00:00","2023-09-07 17:11:48.100000+00:00","wftech-camera-00108","wftech-camera-00107"],
        ["dahlia","2023-09-07 17:12:28.800000+00:00","2023-09-07 17:12:34.700000+00:00","wftech-camera-00100","wftech-camera-00101"],
        ["dahlia","2023-09-07 17:30:54.100000+00:00","2023-09-07 17:31:03.100000+00:00","wftech-camera-00100","wftech-camera-00100"],
        ["dahlia","2023-09-07 17:31:08.500000+00:00","2023-09-07 17:31:30.700000+00:00","wftech-camera-00100","wftech-camera-00106"],
        ["dahlia","2023-09-07 17:31:32.900000+00:00","2023-09-07 17:31:36.700000+00:00","wftech-camera-00106","wftech-camera-00106"],
        ["dahlia","2023-09-07 17:31:39.500000+00:00","2023-09-07 17:31:41.600000+00:00","wftech-camera-00107","wftech-camera-00107"],
        ["dahlia","2023-09-07 17:34:20.800000+00:00","2023-09-07 17:34:40+00:00","wftech-camera-00106","wftech-camera-00111"],
        ["dahlia","2023-09-07 17:37:50.700000+00:00","2023-09-07 17:37:52.600000+00:00","wftech-camera-00100","wftech-camera-00100"],
        ["dahlia","2023-09-07 17:43:12.300000+00:00","2023-09-07 17:43:18.700000+00:00","wftech-camera-00100","wftech-camera-00100"],
        ["dahlia","2023-09-07 17:44:09+00:00","2023-09-07 17:44:15.100000+00:00","wftech-camera-00100","wftech-camera-00102"],
        ["dahlia","2023-09-07 17:47:51.400000+00:00","2023-09-07 17:48:05.400000+00:00","wftech-camera-00107","wftech-camera-00111"],
        ["dahlia","2023-09-07 17:56:35.300000+00:00","2023-09-07 17:56:51+00:00","wftech-camera-00100","wftech-camera-00102"],
        ["dahlia","2023-09-07 17:56:39.300000+00:00","2023-09-07 17:56:45.400000+00:00","wftech-camera-00112","wftech-camera-00101"],
        ["dahlia","2023-09-07 17:56:47.100000+00:00","2023-09-07 17:56:48.300000+00:00","wftech-camera-00101","wftech-camera-00101"],
        ["dahlia","2023-09-07 17:56:50.700000+00:00","2023-09-07 17:56:55.800000+00:00","wftech-camera-00100","wftech-camera-00100"],
        ["dahlia","2023-09-07 17:56:59+00:00","2023-09-07 17:57:04.900000+00:00","wftech-camera-00100","wftech-camera-00100"],
        ["dahlia","2023-09-07 17:57:11.400000+00:00","2023-09-07 17:57:14.800000+00:00","wftech-camera-00101","wftech-camera-00100"],
        ["dahlia","2023-09-07 17:57:24+00:00","2023-09-07 17:57:31+00:00","wftech-camera-00100","wftech-camera-00100"],
        ["dahlia","2023-09-07 17:57:35.500000+00:00","2023-09-07 17:58:02.800000+00:00","wftech-camera-00100","wftech-camera-00102"],
        ["dahlia","2023-09-07 17:58:04.400000+00:00","2023-09-07 17:58:35+00:00","wftech-camera-00102","wftech-camera-00100"],
        ["dahlia","2023-09-07 17:59:09.200000+00:00","2023-09-07 17:59:10.600000+00:00","wftech-camera-00102","wftech-camera-00102"],
        ["dahlia","2023-09-07 17:59:18.900000+00:00","2023-09-07 17:59:21.500000+00:00","wftech-camera-00111","wftech-camera-00111"],
        ["dahlia","2023-09-07 17:59:23.300000+00:00","2023-09-07 17:59:26.900000+00:00","wftech-camera-00111","wftech-camera-00111"],
        ["dahlia","2023-09-07 18:03:38.300000+00:00","2023-09-07 18:03:41.700000+00:00","wftech-camera-00100","wftech-camera-00100"],
        ["dahlia","2023-09-07 18:05:29.600000+00:00","2023-09-07 18:05:38+00:00","wftech-camera-00112","wftech-camera-00112"],
        ["dahlia","2023-09-07 18:05:41.500000+00:00","2023-09-07 18:05:49.900000+00:00","wftech-camera-00112","wftech-camera-00100"],
        ["dahlia","2023-09-07 18:05:57.100000+00:00","2023-09-07 18:06:03.600000+00:00","wftech-camera-00100","wftech-camera-00100"],
        ["dahlia","2023-09-07 18:06:05.200000+00:00","2023-09-07 18:06:16.800000+00:00","wftech-camera-00100","wftech-camera-00112"],
        ["dahlia","2023-09-07 18:07:57.800000+00:00","2023-09-07 18:08:04.800000+00:00","wftech-camera-00100","wftech-camera-00100"],
        ["dahlia","2023-09-07 18:11:36.100000+00:00","2023-09-07 18:11:43.600000+00:00","wftech-camera-00101","wftech-camera-00101"],
        ["dahlia","2023-09-07 18:12:30.400000+00:00","2023-09-07 18:12:38.700000+00:00","wftech-camera-00112","wftech-camera-00100"],
        ["dahlia","2023-09-07 18:12:41+00:00","2023-09-07 18:12:42.100000+00:00","wftech-camera-00100","wftech-camera-00100"],
        ["dahlia","2023-09-07 18:12:43.800000+00:00","2023-09-07 18:12:53.700000+00:00","wftech-camera-00100","wftech-camera-00101"],
        ["dahlia","2023-09-07 18:13:59+00:00","2023-09-07 18:14:00.500000+00:00","wftech-camera-00100","wftech-camera-00100"],
        ["dahlia","2023-09-07 18:15:10.500000+00:00","2023-09-07 18:15:12+00:00","wftech-camera-00101","wftech-camera-00101"],
        ["dahlia","2023-09-07 18:18:02.200000+00:00","2023-09-07 18:18:08+00:00","wftech-camera-00112","wftech-camera-00110"],
        ["dahlia","2023-09-07 18:30:43.400000+00:00","2023-09-07 18:30:49.400000+00:00","wftech-camera-00100","wftech-camera-00100"],
        ["dahlia","2023-09-07 18:31:03.800000+00:00","2023-09-07 18:31:13.400000+00:00","wftech-camera-00101","wftech-camera-00109"],
        ["dahlia","2023-09-07 18:49:41.300000+00:00","2023-09-07 18:49:55+00:00","wftech-camera-00105","wftech-camera-00110"],
        ["dahlia","2023-09-07 19:10:03.300000+00:00","2023-09-07 19:10:11.100000+00:00","wftech-camera-00100","wftech-camera-00102"],
        ["dahlia","2023-09-07 20:06:52.400000+00:00","2023-09-07 20:07:07.400000+00:00","wftech-camera-00111","wftech-camera-00105"],
        ["dahlia","2023-09-07 20:19:46.900000+00:00","2023-09-07 20:19:52+00:00","wftech-camera-00105","wftech-camera-00105"],
        ["dahlia","2023-09-07 20:19:56.100000+00:00","2023-09-07 20:19:57.100000+00:00","wftech-camera-00105","wftech-camera-00105"],
        ["dahlia","2023-09-07 20:20:12.600000+00:00","2023-09-07 20:20:22.200000+00:00","wftech-camera-00100","wftech-camera-00102"],
        ["dahlia","2023-09-07 20:29:32.100000+00:00","2023-09-07 20:29:35.600000+00:00","wftech-camera-00111","wftech-camera-00111"],
        ["dahlia","2023-09-07 20:29:37.700000+00:00","2023-09-07 20:29:51.100000+00:00","wftech-camera-00111","wftech-camera-00111"],
        ["dahlia","2023-09-07 20:32:51.500000+00:00","2023-09-07 20:33:24.100000+00:00","wftech-camera-00100","wftech-camera-00109"],
        ["dahlia","2023-09-07 20:47:07.800000+00:00","2023-09-07 20:47:08.800000+00:00","wftech-camera-00100","wftech-camera-00100"],
        ["dahlia","2023-09-07 20:48:13.100000+00:00","2023-09-07 20:48:19+00:00","wftech-camera-00101","wftech-camera-00101"],
        ["dahlia","2023-09-07 20:48:22.100000+00:00","2023-09-07 20:48:28.600000+00:00","wftech-camera-00101","wftech-camera-00101"],
        ["dahlia","2023-09-07 20:50:28.800000+00:00","2023-09-07 20:50:30.100000+00:00","wftech-camera-00100","wftech-camera-00100"],
        ["dahlia","2023-09-07 20:50:31.300000+00:00","2023-09-07 20:50:40.900000+00:00","wftech-camera-00100","wftech-camera-00105"],
        ["dahlia","2023-09-07 20:52:30.800000+00:00","2023-09-07 20:52:32.300000+00:00","wftech-camera-00111","wftech-camera-00111"],
        ["dahlia","2023-09-07 20:53:39.600000+00:00","2023-09-07 20:53:46.400000+00:00","wftech-camera-00100","wftech-camera-00112"],
        ["dahlia","2023-09-07 20:54:01.900000+00:00","2023-09-07 20:54:05.600000+00:00","wftech-camera-00105","wftech-camera-00105"],
        ["dahlia","2023-09-07 20:57:10+00:00","2023-09-07 20:57:13.900000+00:00","wftech-camera-00110","wftech-camera-00100"],
        ["dahlia","2023-09-07 21:05:38.900000+00:00","2023-09-07 21:05:48.800000+00:00","wftech-camera-00102","wftech-camera-00111"],
        ["dahlia","2023-09-07 21:14:01.900000+00:00","2023-09-07 21:14:08.200000+00:00","wftech-camera-00100","wftech-camera-00111"],
        ["dahlia","2023-09-07 21:21:00.100000+00:00","2023-09-07 21:21:09.200000+00:00","wftech-camera-00101","wftech-camera-00102"],
        ["dahlia","2023-09-07 21:22:50+00:00","2023-09-07 21:24:28.900000+00:00","wftech-camera-00101","wftech-camera-00106"],
        ["dahlia","2023-09-07 21:26:22.600000+00:00","2023-09-07 21:26:26.500000+00:00","wftech-camera-00102","wftech-camera-00102"],
        ["dahlia","2023-09-07 21:27:54.500000+00:00","2023-09-07 21:27:55.900000+00:00","wftech-camera-00108","wftech-camera-00108"],
        ["dahlia","2023-09-07 21:27:57.800000+00:00","2023-09-07 21:28:07.100000+00:00","wftech-camera-00108","wftech-camera-00109"],
        ["dahlia","2023-09-07 21:28:09.700000+00:00","2023-09-07 21:28:19.500000+00:00","wftech-camera-00109","wftech-camera-00101"],
        ["dahlia","2023-09-07 21:34:18.400000+00:00","2023-09-07 21:34:19.700000+00:00","wftech-camera-00102","wftech-camera-00102"],
        ["dahlia","2023-09-07 21:45:32.800000+00:00","2023-09-07 21:45:42.400000+00:00","wftech-camera-00110","wftech-camera-00111"],
        ["dahlia","2023-09-07 21:45:52.600000+00:00","2023-09-07 21:46:04.100000+00:00","wftech-camera-00110","wftech-camera-00111"],
        ["dahlia","2023-09-07 21:47:20+00:00","2023-09-07 21:47:40+00:00","wftech-camera-00100","wftech-camera-00101"],
        ["dahlia","2023-09-07 21:51:44.500000+00:00","2023-09-07 21:52:03.300000+00:00","wftech-camera-00105","wftech-camera-00111"],
        ["dahlia","2023-09-07 22:20:41.700000+00:00","2023-09-07 22:20:43.900000+00:00","wftech-camera-00111","wftech-camera-00111"],
        ["dahlia","2023-09-07 22:23:39+00:00","2023-09-07 22:23:50.500000+00:00","wftech-camera-00100","wftech-camera-00105"],
        ["dahlia","2023-09-07 22:24:04.500000+00:00","2023-09-07 22:24:10.700000+00:00","wftech-camera-00100","wftech-camera-00112"],
        ["dahlia","2023-09-07 22:56:31.500000+00:00","2023-09-07 22:56:39.900000+00:00","wftech-camera-00111","wftech-camera-00100"],
        ["dahlia","2023-09-07 22:56:42.800000+00:00","2023-09-07 22:56:50.100000+00:00","wftech-camera-00111","wftech-camera-00100"],
        ["dahlia","2023-09-07 22:56:51.500000+00:00","2023-09-07 22:56:54.600000+00:00","wftech-camera-00100","wftech-camera-00100"],
        ["dahlia","2023-09-07 22:56:57.600000+00:00","2023-09-07 22:57:05.400000+00:00","wftech-camera-00111","wftech-camera-00100"],
        ["dahlia","2023-09-07 22:57:10.400000+00:00","2023-09-07 22:57:13.200000+00:00","wftech-camera-00111","wftech-camera-00111"],
        ["dahlia","2023-09-07 22:57:15.200000+00:00","2023-09-07 22:57:18.600000+00:00","wftech-camera-00111","wftech-camera-00111"],
        ["dahlia","2023-09-07 22:57:21.500000+00:00","2023-09-07 22:57:36+00:00","wftech-camera-00100","wftech-camera-00100"],
        ["dahlia","2023-09-07 22:57:49.300000+00:00","2023-09-07 22:57:52.600000+00:00","wftech-camera-00111","wftech-camera-00111"]
    ]
    events_start_and_end_separated = list(np.array(events_list)[:, [0, 1, 3]]) + list(
        np.array(events_list)[:, [0, 2, 4]]
    )

    events = list(
        map(
            lambda e: {
                "environment": e[0],
                "start": datetime.fromisoformat(e[1]),
                "end": datetime.fromisoformat(e[1]),
                "cameras": e[2],
            },
            events_start_and_end_separated,
        )
    )

    def _overlay(event):
        logging.info(
            f"Generating overlays for, Environment: {event['environment']} Start: {event['start']} End: {event['end']} Cameras: {event['cameras']}"
        )
        overlay_all_geoms_on_all_video_for_given_time(
            c=event["environment"],
            s=event["start"],
            e=event["end"],
            cameras=event["cameras"],
            video_start_end_seconds_offset=3,
        )

    with multiprocessing.pool.ThreadPool(processes=2) as pool:
        pool.map(_overlay, events)

        pool.close()
        pool.join()

    # generate_video_snippets_for_person(
    #     classroom_name="dahlia",
    #     person_device_id="569d24ba-021f-4dcd-a059-f2b899752f07",
    #     start=datetime.strptime("2023-06-30T12:40:18.771697-0700", "%Y-%m-%dT%H:%M:%S.%f%z"),
    #     end=datetime.strptime("2023-06-30T12:43:24.032254-0700", "%Y-%m-%dT%H:%M:%S.%f%z"),
    # )
