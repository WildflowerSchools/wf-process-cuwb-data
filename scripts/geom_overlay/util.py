import functools
import logging
import multiprocessing
import os
import tempfile
from datetime import timedelta, datetime
from functools import partial
from pathlib import Path

import cv_utils
import ffmpeg
import honeycomb_io
import numpy as np
import pandas as pd
import video_io
from geom_render import GeomCollection2D

from process_cuwb_data import fetch_geoms_2d, HoneycombCachingClient, CameraUWBLineOfSight

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


def overlay_all_geoms_on_all_video_for_given_time(
    c, s, e, cameras=None, device_ids=None, video_start_end_seconds_offset=0
):
    environment_id = honeycomb_io.fetch_environment_id(
        environment_name=c,
    )

    geoms = fetch_geoms_2d(
        environment_name=c,
        start_time=s - timedelta(seconds=video_start_end_seconds_offset),
        end_time=e + timedelta(seconds=video_start_end_seconds_offset),
        smooth=True,
        device_ids=device_ids,
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
