import datetime
import os

import pandas as pd
import numpy as np

import geom_render
from honeycomb_io import fetch_camera_info, fetch_camera_calibrations

from process_cuwb_data.honeycomb_imu_data import fetch_imu_data, smooth_imu_position_data
from process_cuwb_data.utils.log import logger


def fetch_geoms_2d(
    environment_name,
    start_time,
    end_time,
    df_cuwb_position_data=None,
    z_axis_override=0.5,
    device_ids=None,
    smooth=False,
    frames_per_second=10.0,
    progress_bar=False,
    notebook=False,
):
    # Fetch CUWB position data
    if df_cuwb_position_data is not None:
        df_position = df_cuwb_position_data.loc[
            (df_cuwb_position_data.index >= start_time) & (df_cuwb_position_data.index <= end_time)
        ]
    else:
        df_position = fetch_imu_data(
            imu_type="position",
            environment_name=environment_name,
            start=start_time,
            end=end_time,
            device_ids=device_ids,
        )

    df_position = df_position[
        [
            "x",
            "y",
            "z",
            "assignment_id",
            "device_id",
            "device_tag_id",
            "device_name",
            "device_serial_number",
            "entity_type",
            "person_id",
            "person_short_name",
            "person_anonymized_short_name",
            "material_id",
            "material_assignment_id",
            "material_name",
        ]
    ]

    if z_axis_override:
        df_position["z"] = z_axis_override

    if device_ids:
        df_position = df_position[df_position["device_id"].isin(device_ids)]

    if smooth:
        df_position = smooth_imu_position_data(df_position=df_position)

    # Create 3D geom collection from data
    geom_collection_3d = create_geom_collection_3d(
        df_position,
        progress_bar=progress_bar,
        notebook=notebook,
        start_time=start_time,
        end_time=end_time,
        frames_per_second=frames_per_second,
    )

    # Fetch camera info
    camera_info_df = fetch_camera_info(environment_name=environment_name, start=start_time, end=end_time)
    camera_calibrations = fetch_camera_calibrations(
        camera_ids=camera_info_df.index.unique().to_list(), start=start_time, end=end_time
    )
    camera_calibrations_df = pd.DataFrame.from_dict(camera_calibrations, orient="index")
    camera_info_with_calibrations_df = camera_calibrations_df.join(camera_info_df)

    # Project onto camera views
    geom_collection_2d_dict = project_onto_camera_views(
        geom_3d=geom_collection_3d, camera_info_df=camera_info_with_calibrations_df
    )
    return geom_collection_2d_dict


def create_geom_collection_3d(
    df,
    start_time,
    end_time,
    frames_per_second=10.0,
    colors=None,
    progress_bar=False,
    notebook=False,
):
    # Create dictionary of 3D geom collections, one for each object in data
    if colors is None:
        colors = {"Person": "#ff0000", "Tray": "#00ff00"}

    logger.info(
        "Creating dictionary of 3D geom collections for each sensor in data: {}".format(
            df["device_serial_number"].unique().tolist()
        )
    )

    df = df[
        [
            "x",
            "y",
            "z",
            "assignment_id",
            "device_id",
            "device_tag_id",
            "device_name",
            "device_serial_number",
            "entity_type",
            "person_id",
            "person_short_name",
            "person_anonymized_short_name",
            "material_id",
            "material_assignment_id",
            "material_name",
        ]
    ]
    geom_collection_3d_dict = {}
    for (
        device_id,
        device_serial_number,
        entity_type,
        person_id,
        person_anonymized_short_name,
        material_id,
        material_name,
    ), group_df in df.fillna("NA").groupby(
        [
            "device_id",
            "device_serial_number",
            "entity_type",
            "person_id",
            "person_anonymized_short_name",
            "material_id",
            "material_name",
        ]
    ):
        entity_name = material_name
        if entity_type == "Person":
            entity_name = person_anonymized_short_name
        entity_id = material_id
        if entity_type == "Person":
            entity_id = person_id
        logger.info(
            "Creating 3D geom collection for {} ({}) [{} to {}]".format(
                entity_name, device_serial_number, group_df.index.min().isoformat(), group_df.index.max().isoformat()
            )
        )
        time_index = group_df.index.to_pydatetime()

        position_values = group_df.loc[:, ["x", "y", "z"]].values

        coordinates = np.expand_dims(position_values, axis=1)
        if position_values.shape[0] != time_index.shape[0]:
            coordinates = np.resize(coordinates, (time_index.shape[0], 1, 3))

        geom_list = [
            geom_render.Point3D(
                coordinate_indices=[0],
                color=colors[entity_type] if entity_type in colors else "#0000ff",
                object_type=entity_type,
                object_id=entity_id,
                object_name=entity_name,
                time_index=time_index,
            ),
            geom_render.Text3D(
                text=entity_name,
                coordinate_indices=[0],
                color=colors[entity_type] if entity_type in colors else "#0000ff",
                object_type=entity_type,
                object_id=entity_id,
                object_name=entity_name,
                time_index=time_index,
            ),
        ]
        geom_collection_3d_dict[device_id] = geom_render.GeomCollection3D(
            time_index=time_index, coordinates=coordinates, geom_list=geom_list
        )
    # Create combined 3D geom collection
    logger.info("Combining 3D geom collections into single 3D geom collection")
    combined_geom_collection_3d = geom_render.GeomCollection3D.from_geom_list(
        list(geom_collection_3d_dict.values()),
        start_time,
        end_time,
        frames_per_second,
        progress_bar=progress_bar,
        notebook=notebook,
    )
    return combined_geom_collection_3d


def resample_geom(geom, start_time, end_time, frames_per_second=10.0, progress_bar=False, notebook=False):
    logger.info(
        "Resampling geom to match start time {}, end time {}, and {} frames per second".format(
            start_time.isoformat(), end_time.isoformat(), frames_per_second
        )
    )
    time_between_frames = datetime.timedelta(microseconds=int(round(10**6 / frames_per_second)))
    num_frames = int(round((end_time - start_time) / time_between_frames))
    geom_resampled = geom.resample(
        new_start_time=start_time,
        new_frames_per_second=frames_per_second,
        new_num_frames=num_frames,
        progress_bar=progress_bar,
        notebook=notebook,
    )
    return geom_resampled


def project_onto_camera_views(geom_3d, camera_info_df):
    logger.info(f"Creating 2D geoms from 3D geom, one for each camera: {camera_info_df['device_name'].to_dict()}")
    geom_2d_dict = {}
    for device_id, camera_info in camera_info_df.iterrows():
        logger.info(f"Creating 2D geom for camera {camera_info['device_name']}")
        geom_2d_dict[device_id] = {}
        geom_2d_dict[device_id]["device_name"] = camera_info["device_name"]
        geom_2d_dict[device_id]["geom"] = geom_3d.project(
            rotation_vector=camera_info["rotation_vector"],
            translation_vector=camera_info["translation_vector"],
            camera_matrix=camera_info["camera_matrix"],
            distortion_coefficients=camera_info["distortion_coefficients"],
            frame_width=camera_info["image_width"],
            frame_height=camera_info["image_height"],
        )
    return geom_2d_dict


def write_json(geom_dict, output_directory=".", prefix="geom_2d", indent=None):
    logger.info(
        f"Writing geom data to local JSON file for: {[geom_info['device_name'] for geom_info in geom_dict.values()]}"
    )
    for device_id, geom_info in geom_dict.items():
        logger.info(f"Writing geom data to local JSON file for {geom_info['device_name']}")
        path = os.path.join(output_directory, "_".join([prefix, geom_info["device_name"]]) + ".json")
        with open(path, "w", encoding="UTF-8") as fp:
            fp.write(geom_info["geom"].to_json(indent=indent))
