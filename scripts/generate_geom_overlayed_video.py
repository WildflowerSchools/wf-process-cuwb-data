import functools
import tempfile
from collections import OrderedDict
from datetime import datetime, timedelta
from functools import partial
import logging
import math
import multiprocessing
import os
import sys
import time
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
    output_overlayed_dir = "./output/overlayed_video"
    output_concatenated_videos_dir = "./output/concatenated_videos"
    output_raw_video_snippet_dir = "../output/raw_videos"

    environment_id = honeycomb_io.fetch_environment_id(
        environment_name=c,
    )

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
        output_directory=output_concatenated_videos_dir,
        video_snippet_directory=output_raw_video_snippet_dir,
    )
    if df_video_snippets is None:
        logging.error(f"Missing video for {c} - from {s} - {e}")
        return

    df_video_outputs = df_video_snippets.copy()

    def generate_output_file_names(x):
        return f"{output_overlayed_dir}/{c}_{s.strftime('%m%d%YT%H%M%S')}_{e.strftime('%m%d%YT%H%M%S')}_{x['camera_device_id']}.mp4"

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
    #
    # for video_output in df_video_outputs:
    #     logging.info(f"Video create: {video_output['overlayed_file_path']}")


def generate_video_snippets_for_person(
    classroom_name: str,
    person_device_id: str,
    start: datetime,
    end: datetime,
):

    output_overlayed_dir = "./output/overlayed_video"
    output_final_overlayed_dir = "./output/final_overlayed_video"
    output_concatenated_videos_dir = "./output/concatenated_videos"
    output_raw_video_snippet_dir = "../output/raw_videos"

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
            output_directory=output_concatenated_videos_dir,
            video_snippet_directory=output_raw_video_snippet_dir,
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
        output_path = f"{output_overlayed_dir}/{file_path_obj.stem}_overlayed{file_path_obj.suffix}"

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
    video_concatenated_output_path = f"{output_concatenated_videos_dir}/{classroom_name}_{person_device_id}_{start.strftime('%m%d%YT%H%M%S%z')}_{end.strftime('%m%d%YT%H%M%S%z')}.mp4"

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

    final_overlayed_video_output_path = f"{output_final_overlayed_dir}/{classroom_name}_{person_device_id}_{start.strftime('%m%d%YT%H%M%S%z')}_{end.strftime('%m%d%YT%H%M%S%z')}.mp4"
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
        [
            "dahlia",
            "2023-08-29T15:36:22.000000+00:00",
            "2023-08-29T15:36:30.800000+00:00",
            "dc4168df-7d67-4589-82cb-b4fc2d212977",
            "wftech-camera-00105",
            "wftech-camera-00105",
        ],
        [
            "dahlia",
            "2023-08-29T15:38:04.200000+00:00",
            "2023-08-29T15:38:05.600000+00:00",
            "9de414fd-f1e4-495f-8410-f4f9bab2b401",
            "wftech-camera-00111",
            "wftech-camera-00111",
        ],
        [
            "dahlia",
            "2023-08-29T16:01:45.600000+00:00",
            "2023-08-29T16:02:00.400000+00:00",
            "e0b6754a-9171-4546-b19d-8cd2db65cb97",
            "wftech-camera-00100",
            "wftech-camera-00100",
        ],
        [
            "dahlia",
            "2023-08-29T16:03:08.500000+00:00",
            "2023-08-29T16:03:18.400000+00:00",
            "e0b6754a-9171-4546-b19d-8cd2db65cb97",
            "wftech-camera-00100",
            "wftech-camera-00100",
        ],
        [
            "dahlia",
            "2023-08-29T16:04:31.600000+00:00",
            "2023-08-29T16:04:44.500000+00:00",
            "e0b6754a-9171-4546-b19d-8cd2db65cb97",
            "wftech-camera-00112",
            "wftech-camera-00109",
        ],
        [
            "dahlia",
            "2023-08-29T16:06:09.600000+00:00",
            "2023-08-29T16:06:11.100000+00:00",
            "5c94cc5a-11d9-4636-81be-467a6f0357e5",
            "wftech-camera-00111",
            "wftech-camera-00111",
        ],
        [
            "dahlia",
            "2023-08-29T16:09:28.600000+00:00",
            "2023-08-29T16:09:32.800000+00:00",
            "57caaf6b-54d3-4cb6-8626-7612fa3ae57b",
            "wftech-camera-00101",
            "wftech-camera-00101",
        ],
        [
            "dahlia",
            "2023-08-29T16:09:34.900000+00:00",
            "2023-08-29T16:09:38.600000+00:00",
            "57caaf6b-54d3-4cb6-8626-7612fa3ae57b",
            "wftech-camera-00101",
            "wftech-camera-00101",
        ],
        [
            "dahlia",
            "2023-08-29T16:10:20.300000+00:00",
            "2023-08-29T16:10:21.700000+00:00",
            "e0b6754a-9171-4546-b19d-8cd2db65cb97",
            "wftech-camera-00109",
            "wftech-camera-00109",
        ],
        [
            "dahlia",
            "2023-08-29T16:10:25.700000+00:00",
            "2023-08-29T16:10:27.700000+00:00",
            "e0b6754a-9171-4546-b19d-8cd2db65cb97",
            "wftech-camera-00109",
            "wftech-camera-00109",
        ],
        [
            "dahlia",
            "2023-08-29T16:16:10.000000+00:00",
            "2023-08-29T16:16:21.200000+00:00",
            "e0b6754a-9171-4546-b19d-8cd2db65cb97",
            "wftech-camera-00102",
            "wftech-camera-00110",
        ],
        [
            "dahlia",
            "2023-08-29T16:27:49.000000+00:00",
            "2023-08-29T16:27:55.900000+00:00",
            "5c94cc5a-11d9-4636-81be-467a6f0357e5",
            "wftech-camera-00100",
            "wftech-camera-00100",
        ],
        [
            "dahlia",
            "2023-08-29T16:34:55.700000+00:00",
            "2023-08-29T16:35:05.500000+00:00",
            "e6827e56-1c72-41c9-8fd5-dfcd9b650d78",
            "wftech-camera-00112",
            "wftech-camera-00100",
        ],
        [
            "dahlia",
            "2023-08-29T16:37:30.100000+00:00",
            "2023-08-29T16:37:31.200000+00:00",
            "e6827e56-1c72-41c9-8fd5-dfcd9b650d78",
            "wftech-camera-00100",
            "wftech-camera-00100",
        ],
        [
            "dahlia",
            "2023-08-29T16:40:17.400000+00:00",
            "2023-08-29T16:40:19.200000+00:00",
            "b08d7e80-f6b1-4cfa-974a-74447677d57f",
            "wftech-camera-00102",
            "wftech-camera-00102",
        ],
        [
            "dahlia",
            "2023-08-29T16:41:05.300000+00:00",
            "2023-08-29T16:41:06.900000+00:00",
            "e6827e56-1c72-41c9-8fd5-dfcd9b650d78",
            "wftech-camera-00100",
            "wftech-camera-00100",
        ],
        [
            "dahlia",
            "2023-08-29T16:47:10.500000+00:00",
            "2023-08-29T16:47:24.700000+00:00",
            "0ce5935a-1d67-4729-b5e4-88ea56d545c8",
            "wftech-camera-00111",
            "wftech-camera-00111",
        ],
        [
            "dahlia",
            "2023-08-29T16:53:29.700000+00:00",
            "2023-08-29T16:53:36.200000+00:00",
            "5c94cc5a-11d9-4636-81be-467a6f0357e5",
            "wftech-camera-00101",
            "wftech-camera-00111",
        ],
        [
            "dahlia",
            "2023-08-29T16:55:49.100000+00:00",
            "2023-08-29T16:55:50.600000+00:00",
            "b08d7e80-f6b1-4cfa-974a-74447677d57f",
            "wftech-camera-00102",
            "wftech-camera-00102",
        ],
        [
            "dahlia",
            "2023-08-29T16:56:16.300000+00:00",
            "2023-08-29T16:56:17.600000+00:00",
            "dc4168df-7d67-4589-82cb-b4fc2d212977",
            "wftech-camera-00105",
            "wftech-camera-00105",
        ],
        [
            "dahlia",
            "2023-08-29T16:59:42.100000+00:00",
            "2023-08-29T16:59:52.400000+00:00",
            "e0b6754a-9171-4546-b19d-8cd2db65cb97",
            "wftech-camera-00105",
            "wftech-camera-00110",
        ],
        [
            "dahlia",
            "2023-08-29T17:15:06.100000+00:00",
            "2023-08-29T17:15:08.600000+00:00",
            "e6827e56-1c72-41c9-8fd5-dfcd9b650d78",
            "wftech-camera-00100",
            "wftech-camera-00100",
        ],
        [
            "dahlia",
            "2023-08-29T17:15:09.900000+00:00",
            "2023-08-29T17:15:11.500000+00:00",
            "e6827e56-1c72-41c9-8fd5-dfcd9b650d78",
            "wftech-camera-00100",
            "wftech-camera-00100",
        ],
        [
            "dahlia",
            "2023-08-29T17:15:13.600000+00:00",
            "2023-08-29T17:15:15.400000+00:00",
            "e6827e56-1c72-41c9-8fd5-dfcd9b650d78",
            "wftech-camera-00100",
            "wftech-camera-00100",
        ],
        [
            "dahlia",
            "2023-08-29T17:15:23.700000+00:00",
            "2023-08-29T17:15:25.300000+00:00",
            "e6827e56-1c72-41c9-8fd5-dfcd9b650d78",
            "wftech-camera-00100",
            "wftech-camera-00100",
        ],
        [
            "dahlia",
            "2023-08-29T17:15:30.000000+00:00",
            "2023-08-29T17:15:30.900000+00:00",
            "e6827e56-1c72-41c9-8fd5-dfcd9b650d78",
            "wftech-camera-00100",
            "wftech-camera-00100",
        ],
        [
            "dahlia",
            "2023-08-29T17:15:33.900000+00:00",
            "2023-08-29T17:15:39.100000+00:00",
            "e6827e56-1c72-41c9-8fd5-dfcd9b650d78",
            "wftech-camera-00100",
            "wftech-camera-00100",
        ],
        [
            "dahlia",
            "2023-08-29T17:16:03.600000+00:00",
            "2023-08-29T17:16:05.800000+00:00",
            "e6827e56-1c72-41c9-8fd5-dfcd9b650d78",
            "wftech-camera-00100",
            "wftech-camera-00100",
        ],
        [
            "dahlia",
            "2023-08-29T17:18:40.400000+00:00",
            "2023-08-29T17:18:43.000000+00:00",
            "e6827e56-1c72-41c9-8fd5-dfcd9b650d78",
            "wftech-camera-00100",
            "wftech-camera-00100",
        ],
        [
            "dahlia",
            "2023-08-29T17:33:18.200000+00:00",
            "2023-08-29T17:33:21.300000+00:00",
            "e6827e56-1c72-41c9-8fd5-dfcd9b650d78",
            "wftech-camera-00112",
            "wftech-camera-00112",
        ],
        [
            "dahlia",
            "2023-08-29T17:55:42.500000+00:00",
            "2023-08-29T17:55:47.300000+00:00",
            "5c94cc5a-11d9-4636-81be-467a6f0357e5",
            "wftech-camera-00100",
            "wftech-camera-00100",
        ],
        [
            "dahlia",
            "2023-08-29T17:55:49.500000+00:00",
            "2023-08-29T17:55:51.100000+00:00",
            "5c94cc5a-11d9-4636-81be-467a6f0357e5",
            "wftech-camera-00100",
            "wftech-camera-00100",
        ],
        [
            "dahlia",
            "2023-08-29T17:57:02.700000+00:00",
            "2023-08-29T18:12:35.400000+00:00",
            "40ae8262-9777-4650-aeed-b7e6b435dff0",
            "wftech-camera-00101",
            "wftech-camera-00112",
        ],
        [
            "dahlia",
            "2023-08-29T18:06:04.500000+00:00",
            "2023-08-29T18:06:24.900000+00:00",
            "b08d7e80-f6b1-4cfa-974a-74447677d57f",
            "wftech-camera-00100",
            "wftech-camera-00106",
        ],
        [
            "dahlia",
            "2023-08-29T18:06:28.100000+00:00",
            "2023-08-29T18:06:30.600000+00:00",
            "b08d7e80-f6b1-4cfa-974a-74447677d57f",
            "wftech-camera-00106",
            "wftech-camera-00106",
        ],
        [
            "dahlia",
            "2023-08-29T18:09:58.400000+00:00",
            "2023-08-29T18:10:14.900000+00:00",
            "b08d7e80-f6b1-4cfa-974a-74447677d57f",
            "wftech-camera-00106",
            "wftech-camera-00106",
        ],
        [
            "dahlia",
            "2023-08-29T18:10:16.700000+00:00",
            "2023-08-29T18:10:19.100000+00:00",
            "b08d7e80-f6b1-4cfa-974a-74447677d57f",
            "wftech-camera-00106",
            "wftech-camera-00106",
        ],
        [
            "dahlia",
            "2023-08-29T18:10:21.100000+00:00",
            "2023-08-29T18:10:22.400000+00:00",
            "b08d7e80-f6b1-4cfa-974a-74447677d57f",
            "wftech-camera-00106",
            "wftech-camera-00106",
        ],
        [
            "dahlia",
            "2023-08-29T18:10:26.400000+00:00",
            "2023-08-29T18:10:58.900000+00:00",
            "b08d7e80-f6b1-4cfa-974a-74447677d57f",
            "wftech-camera-00106",
            "wftech-camera-00112",
        ],
        [
            "dahlia",
            "2023-08-29T18:11:50.800000+00:00",
            "2023-08-29T18:11:52.100000+00:00",
            "b08d7e80-f6b1-4cfa-974a-74447677d57f",
            "wftech-camera-00102",
            "wftech-camera-00102",
        ],
        [
            "dahlia",
            "2023-08-29T18:12:36.500000+00:00",
            "2023-08-29T18:12:53.300000+00:00",
            "40ae8262-9777-4650-aeed-b7e6b435dff0",
            "wftech-camera-00102",
            "wftech-camera-00112",
        ],
        [
            "dahlia",
            "2023-08-29T18:12:42.200000+00:00",
            "2023-08-29T18:12:53.800000+00:00",
            "b08d7e80-f6b1-4cfa-974a-74447677d57f",
            "wftech-camera-00100",
            "wftech-camera-00112",
        ],
        [
            "dahlia",
            "2023-08-29T18:12:55.900000+00:00",
            "2023-08-29T18:13:13.300000+00:00",
            "40ae8262-9777-4650-aeed-b7e6b435dff0",
            "wftech-camera-00112",
            "wftech-camera-00101",
        ],
        [
            "dahlia",
            "2023-08-29T18:18:12.900000+00:00",
            "2023-08-29T18:18:38.500000+00:00",
            "9de414fd-f1e4-495f-8410-f4f9bab2b401",
            "wftech-camera-00111",
            "wftech-camera-00109",
        ],
        [
            "dahlia",
            "2023-08-29T18:18:36.500000+00:00",
            "2023-08-29T18:18:39.300000+00:00",
            "5c94cc5a-11d9-4636-81be-467a6f0357e5",
            "wftech-camera-00111",
            "wftech-camera-00111",
        ],
        [
            "dahlia",
            "2023-08-29T18:19:58.200000+00:00",
            "2023-08-29T18:19:59.500000+00:00",
            "9de414fd-f1e4-495f-8410-f4f9bab2b401",
            "wftech-camera-00109",
            "wftech-camera-00109",
        ],
        [
            "dahlia",
            "2023-08-29T18:20:38.300000+00:00",
            "2023-08-29T18:21:03.300000+00:00",
            "b08d7e80-f6b1-4cfa-974a-74447677d57f",
            "wftech-camera-00112",
            "wftech-camera-00106",
        ],
        [
            "dahlia",
            "2023-08-29T18:24:56.700000+00:00",
            "2023-08-29T18:24:58.200000+00:00",
            "b08d7e80-f6b1-4cfa-974a-74447677d57f",
            "wftech-camera-00106",
            "wftech-camera-00106",
        ],
        [
            "dahlia",
            "2023-08-29T18:25:16.200000+00:00",
            "2023-08-29T18:25:17.400000+00:00",
            "9de414fd-f1e4-495f-8410-f4f9bab2b401",
            "wftech-camera-00109",
            "wftech-camera-00109",
        ],
        [
            "dahlia",
            "2023-08-29T18:25:22.800000+00:00",
            "2023-08-29T18:25:24.200000+00:00",
            "b08d7e80-f6b1-4cfa-974a-74447677d57f",
            "wftech-camera-00106",
            "wftech-camera-00106",
        ],
        [
            "dahlia",
            "2023-08-29T18:31:24.900000+00:00",
            "2023-08-29T18:31:27.300000+00:00",
            "b08d7e80-f6b1-4cfa-974a-74447677d57f",
            "wftech-camera-00106",
            "wftech-camera-00106",
        ],
        [
            "dahlia",
            "2023-08-29T18:35:43.000000+00:00",
            "2023-08-29T18:35:45.100000+00:00",
            "b08d7e80-f6b1-4cfa-974a-74447677d57f",
            "wftech-camera-00106",
            "wftech-camera-00106",
        ],
        [
            "dahlia",
            "2023-08-29T18:36:05.900000+00:00",
            "2023-08-29T18:36:08.700000+00:00",
            "b08d7e80-f6b1-4cfa-974a-74447677d57f",
            "wftech-camera-00107",
            "wftech-camera-00106",
        ],
        [
            "dahlia",
            "2023-08-29T18:36:22.600000+00:00",
            "2023-08-29T18:36:28.000000+00:00",
            "b08d7e80-f6b1-4cfa-974a-74447677d57f",
            "wftech-camera-00106",
            "wftech-camera-00106",
        ],
        [
            "dahlia",
            "2023-08-29T18:36:45.600000+00:00",
            "2023-08-29T18:36:47.100000+00:00",
            "b08d7e80-f6b1-4cfa-974a-74447677d57f",
            "wftech-camera-00107",
            "wftech-camera-00107",
        ],
        [
            "dahlia",
            "2023-08-29T18:37:20.500000+00:00",
            "2023-08-29T18:37:22.000000+00:00",
            "b08d7e80-f6b1-4cfa-974a-74447677d57f",
            "wftech-camera-00106",
            "wftech-camera-00106",
        ],
        [
            "dahlia",
            "2023-08-29T18:38:51.000000+00:00",
            "2023-08-29T18:38:52.000000+00:00",
            "9de414fd-f1e4-495f-8410-f4f9bab2b401",
            "wftech-camera-00109",
            "wftech-camera-00109",
        ],
        [
            "dahlia",
            "2023-08-29T18:41:57.500000+00:00",
            "2023-08-29T18:42:04.700000+00:00",
            "b08d7e80-f6b1-4cfa-974a-74447677d57f",
            "wftech-camera-00106",
            "wftech-camera-00106",
        ],
        [
            "dahlia",
            "2023-08-29T18:42:06.900000+00:00",
            "2023-08-29T18:42:08.600000+00:00",
            "b08d7e80-f6b1-4cfa-974a-74447677d57f",
            "wftech-camera-00106",
            "wftech-camera-00106",
        ],
        [
            "dahlia",
            "2023-08-29T18:42:16.900000+00:00",
            "2023-08-29T18:42:18.300000+00:00",
            "b08d7e80-f6b1-4cfa-974a-74447677d57f",
            "wftech-camera-00106",
            "wftech-camera-00106",
        ],
        [
            "dahlia",
            "2023-08-29T18:42:34.800000+00:00",
            "2023-08-29T18:42:36.300000+00:00",
            "b08d7e80-f6b1-4cfa-974a-74447677d57f",
            "wftech-camera-00106",
            "wftech-camera-00106",
        ],
        [
            "dahlia",
            "2023-08-29T18:42:38.100000+00:00",
            "2023-08-29T18:42:40.400000+00:00",
            "b08d7e80-f6b1-4cfa-974a-74447677d57f",
            "wftech-camera-00106",
            "wftech-camera-00106",
        ],
        [
            "dahlia",
            "2023-08-29T18:43:47.100000+00:00",
            "2023-08-29T18:43:51.100000+00:00",
            "b08d7e80-f6b1-4cfa-974a-74447677d57f",
            "wftech-camera-00106",
            "wftech-camera-00106",
        ],
        [
            "dahlia",
            "2023-08-29T18:45:12.800000+00:00",
            "2023-08-29T18:45:25.400000+00:00",
            "b08d7e80-f6b1-4cfa-974a-74447677d57f",
            "wftech-camera-00106",
            "wftech-camera-00106",
        ],
        [
            "dahlia",
            "2023-08-29T18:45:26.700000+00:00",
            "2023-08-29T18:45:28.800000+00:00",
            "b08d7e80-f6b1-4cfa-974a-74447677d57f",
            "wftech-camera-00106",
            "wftech-camera-00106",
        ],
        [
            "dahlia",
            "2023-08-29T18:45:30.800000+00:00",
            "2023-08-29T18:45:32.200000+00:00",
            "b08d7e80-f6b1-4cfa-974a-74447677d57f",
            "wftech-camera-00106",
            "wftech-camera-00106",
        ],
        [
            "dahlia",
            "2023-08-29T18:45:35.100000+00:00",
            "2023-08-29T18:45:39.500000+00:00",
            "b08d7e80-f6b1-4cfa-974a-74447677d57f",
            "wftech-camera-00108",
            "wftech-camera-00108",
        ],
        [
            "dahlia",
            "2023-08-29T18:45:43.200000+00:00",
            "2023-08-29T18:45:44.600000+00:00",
            "b08d7e80-f6b1-4cfa-974a-74447677d57f",
            "wftech-camera-00108",
            "wftech-camera-00108",
        ],
        [
            "dahlia",
            "2023-08-29T18:54:25.000000+00:00",
            "2023-08-29T18:54:48.500000+00:00",
            "b08d7e80-f6b1-4cfa-974a-74447677d57f",
            "wftech-camera-00106",
            "wftech-camera-00102",
        ],
        [
            "dahlia",
            "2023-08-29T18:59:48.000000+00:00",
            "2023-08-29T18:59:53.600000+00:00",
            "9de414fd-f1e4-495f-8410-f4f9bab2b401",
            "wftech-camera-00100",
            "wftech-camera-00110",
        ],
        [
            "dahlia",
            "2023-08-29T21:01:33.300000+00:00",
            "2023-08-29T21:01:39.500000+00:00",
            "4bcc90a5-d4de-4d66-bb14-ad45b9dafe77",
            "wftech-camera-00102",
            "wftech-camera-00109",
        ],
        [
            "dahlia",
            "2023-08-29T21:02:01.500000+00:00",
            "2023-08-29T21:02:05.900000+00:00",
            "4bcc90a5-d4de-4d66-bb14-ad45b9dafe77",
            "wftech-camera-00112",
            "wftech-camera-00112",
        ],
        [
            "dahlia",
            "2023-08-29T21:09:48.100000+00:00",
            "2023-08-29T21:09:49.600000+00:00",
            "dc4168df-7d67-4589-82cb-b4fc2d212977",
            "wftech-camera-00105",
            "wftech-camera-00105",
        ],
        [
            "dahlia",
            "2023-08-29T21:10:09.900000+00:00",
            "2023-08-29T21:10:13.800000+00:00",
            "dc4168df-7d67-4589-82cb-b4fc2d212977",
            "wftech-camera-00105",
            "wftech-camera-00105",
        ],
        [
            "dahlia",
            "2023-08-29T21:25:51.300000+00:00",
            "2023-08-29T21:25:59.200000+00:00",
            "4bcc90a5-d4de-4d66-bb14-ad45b9dafe77",
            "wftech-camera-00112",
            "wftech-camera-00100",
        ],
        [
            "dahlia",
            "2023-08-29T21:28:26.000000+00:00",
            "2023-08-29T21:28:28.100000+00:00",
            "4bcc90a5-d4de-4d66-bb14-ad45b9dafe77",
            "wftech-camera-00100",
            "wftech-camera-00100",
        ],
        [
            "dahlia",
            "2023-08-29T21:28:37.100000+00:00",
            "2023-08-29T21:28:38.200000+00:00",
            "4bcc90a5-d4de-4d66-bb14-ad45b9dafe77",
            "wftech-camera-00100",
            "wftech-camera-00100",
        ],
        [
            "dahlia",
            "2023-08-29T21:30:13.900000+00:00",
            "2023-08-29T21:30:21.000000+00:00",
            "4bcc90a5-d4de-4d66-bb14-ad45b9dafe77",
            "wftech-camera-00100",
            "wftech-camera-00100",
        ],
        [
            "dahlia",
            "2023-08-29T21:30:31.700000+00:00",
            "2023-08-29T21:30:35.400000+00:00",
            "4bcc90a5-d4de-4d66-bb14-ad45b9dafe77",
            "wftech-camera-00100",
            "wftech-camera-00100",
        ],
        [
            "dahlia",
            "2023-08-29T22:09:48.600000+00:00",
            "2023-08-29T22:09:50.600000+00:00",
            "b08d7e80-f6b1-4cfa-974a-74447677d57f",
            "wftech-camera-00102",
            "wftech-camera-00102",
        ],
        [
            "dahlia",
            "2023-08-29T22:52:09.700000+00:00",
            "2023-08-29T22:52:24.800000+00:00",
            "3dd42df9-34ac-47d3-80ef-011f787655ec",
            "wftech-camera-00101",
            "wftech-camera-00101",
        ],
        [
            "dahlia",
            "2023-08-29T22:53:52.700000+00:00",
            "2023-08-29T22:53:54.100000+00:00",
            "b08d7e80-f6b1-4cfa-974a-74447677d57f",
            "wftech-camera-00102",
            "wftech-camera-00102",
        ],
        [
            "dahlia",
            "2023-08-29T23:00:09.800000+00:00",
            "2023-08-29T23:00:18.500000+00:00",
            "5c94cc5a-11d9-4636-81be-467a6f0357e5",
            "wftech-camera-00100",
            "wftech-camera-00111",
        ],
        [
            "dahlia",
            "2023-08-29T23:03:27.400000+00:00",
            "2023-08-29T23:03:34.000000+00:00",
            "e0b6754a-9171-4546-b19d-8cd2db65cb97",
            "wftech-camera-00105",
            "wftech-camera-00105",
        ],
        [
            "dahlia",
            "2023-08-29T23:53:18.100000+00:00",
            "2023-08-29T23:53:22.700000+00:00",
            "e0b6754a-9171-4546-b19d-8cd2db65cb97",
            "wftech-camera-00110",
            "wftech-camera-00110",
        ],
    ]
    events_start_and_end_separated = list(np.array(events_list)[:, [0, 1, 4]]) + list(
        np.array(events_list)[:, [0, 2, 5]]
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
