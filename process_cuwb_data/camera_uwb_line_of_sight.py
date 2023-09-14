import datetime

import cv_utils
import numpy as np
import pandas as pd

from .honeycomb_imu_data import fetch_imu_data, smooth_imu_position_data
from .honeycomb_service import HoneycombCachingClient
from .utils.log import logger


class CameraUWBLineOfSight:
    def __init__(
        self,
        timestamp,
        tag_device_id,
        default_camera_device_id,
        environment_id=None,
        environment_name=None,
        camera_device_ids=None,
        camera_calibrations=None,
        position_window_seconds=4,
        z_axis_override=0.5,
        df_cuwb_position_data=None,
    ):
        honeycomb_caching_client = HoneycombCachingClient()

        self.timestamp = timestamp
        self.tag_device_id = tag_device_id
        self.default_camera_device_id = default_camera_device_id

        if camera_calibrations is None:
            if environment_id is None and environment_name is None and camera_device_ids is None:
                raise ValueError(
                    "If camera calibration info is not specified, must specify either camera device IDs or environment_name ID or environment_name name"
                )

        if environment_name is None:
            df_all_environments = honeycomb_caching_client.fetch_all_environments()
            environment_name = df_all_environments[df_all_environments.index == environment_id].iloc[0][
                "environment_name"
            ]

        if camera_device_ids is None:
            camera_info = honeycomb_caching_client.fetch_camera_devices(
                environment_id=environment_id,
                environment_name=environment_name,
                start=timestamp.replace(minute=0, second=0, microsecond=0),
                end=timestamp.replace(minute=59, second=59, microsecond=0),
            )
            camera_device_ids = camera_info.index.unique().tolist()
        if camera_calibrations is None:
            camera_calibrations = honeycomb_caching_client.fetch_camera_calibrations(
                camera_ids=tuple(camera_device_ids),
                start=timestamp.replace(minute=0, second=0, microsecond=0),
                end=timestamp.replace(minute=59, second=59, microsecond=0),
            )
        position_window_start = timestamp
        position_window_end = timestamp + datetime.timedelta(seconds=position_window_seconds)
        if df_cuwb_position_data is not None:
            df_position_data = df_cuwb_position_data.loc[
                (df_cuwb_position_data.index >= position_window_start)
                & (df_cuwb_position_data.index <= position_window_end)
                & (df_cuwb_position_data["type"] == "position")
            ]
        else:
            df_position_data = fetch_imu_data(
                imu_type="position",
                environment_name=environment_name,
                start=position_window_start,
                end=position_window_end,
                device_ids=[tag_device_id],
            )

        if df_position_data is None or len(df_position_data) == 0:
            err = f"Unable to find position data between {position_window_start} and {position_window_end} for device {tag_device_id}, cannot determine best camera views"
            logger.warning(err)
            raise ValueError(err)

        if tag_device_id is not None:
            df_position_data = df_position_data[df_position_data["device_id"] == tag_device_id]

        if df_position_data is None or len(df_position_data) == 0:
            err = f"Unable to find position data between {position_window_start} and {position_window_end} for device {tag_device_id}, cannot determine best camera views"
            logger.warning(err)
            raise ValueError(err)

        df_position_data = smooth_imu_position_data(df_position=df_position_data)

        try:
            np.nanmedian(df_position_data.loc[:, ["x", "y", "z"]].values, axis=0)
        except:
            logger.info(
                f"Bad df_position_data: {len(df_position_data)} Start - {position_window_start} End - {position_window_end}"
            )
            logger.info(f"df_cuwb_position_data: {len(df_cuwb_position_data)}")

        median_position = np.nanmedian(df_position_data.loc[:, ["x", "y", "z"]].values, axis=0)
        if z_axis_override is not None:
            median_position[2] = z_axis_override

        view_data_list = []
        for camera_device_id, camera_calibration in camera_calibrations.items():
            camera_position = cv_utils.extract_camera_position(
                rotation_vector=camera_calibration["rotation_vector"],
                translation_vector=camera_calibration["translation_vector"],
            )
            distance_from_camera = np.linalg.norm(np.subtract(median_position, camera_position))
            projected_position_2d_coordinates = cv_utils.project_points(
                object_points=median_position.reshape((-1, 3)),
                rotation_vector=camera_calibration["rotation_vector"],
                translation_vector=camera_calibration["translation_vector"],
                camera_matrix=camera_calibration["camera_matrix"],
                distortion_coefficients=camera_calibration["distortion_coefficients"],
                remove_behind_camera=True,
                remove_outside_frame=True,
                image_corners=[[0, 0], [camera_calibration["image_width"], camera_calibration["image_height"]]],
            )
            projected_position_2d_coordinates = np.squeeze(projected_position_2d_coordinates)
            if np.all(np.isfinite(projected_position_2d_coordinates)):
                in_frame = True
                distance_from_image_center = np.linalg.norm(
                    np.subtract(
                        projected_position_2d_coordinates,
                        [camera_calibration["image_width"] / 2, camera_calibration["image_height"] / 2],
                    )
                )
                in_middle = (
                    projected_position_2d_coordinates[0] > camera_calibration["image_width"] * (2.5 / 10.0)
                    and projected_position_2d_coordinates[0] < camera_calibration["image_width"] * (7.5 / 10.0)
                    and projected_position_2d_coordinates[1] > camera_calibration["image_height"] * (2.0 / 10.0)
                    and projected_position_2d_coordinates[1] < camera_calibration["image_height"] * (8.0 / 10.0)
                )
            else:
                in_frame = False
                distance_from_image_center = None
                in_middle = False
            view_data_list.append(
                {
                    "camera_device_id": camera_device_id,
                    "position": median_position,
                    "distance_from_camera": distance_from_camera,
                    "projected_position_2d_coordinates": projected_position_2d_coordinates,
                    "distance_from_image_center": distance_from_image_center,
                    "in_frame": in_frame,
                    "in_middle": in_middle,
                }
            )
        df_view_data = pd.DataFrame(view_data_list).set_index("camera_device_id")
        df_view_data = df_view_data.sort_values("distance_from_image_center")
        self.df_view_data = df_view_data

    def all_camera_views(self):
        return self.df_view_data

    def in_frame_camera_views(self):
        return self.df_view_data.loc[self.df_view_data["in_frame"]].sort_values("distance_from_camera")

    def all_in_frame_camera_views_device_ids(self):
        df_in_frame_views = self.in_frame_camera_views()
        return df_in_frame_views.index.to_list()

    def in_middle_camera_views(self):
        return self.df_view_data.loc[self.df_view_data["in_middle"]].sort_values("distance_from_camera")

    def all_in_middle_camera_views_device_ids(self):
        df_in_middle_views = self.in_middle_camera_views()
        return df_in_middle_views.index.to_list()

    def best_camera_view(self):
        if self.df_view_data["in_middle"].any():
            best_camera_view = self.df_view_data.loc[self.df_view_data["in_middle"]]
        elif self.df_view_data["in_frame"].any():
            best_camera_view = self.df_view_data.loc[self.df_view_data["in_frame"]]
        else:
            best_camera_view = pd.DataFrame(
                [
                    {
                        "camera_device_id": self.default_camera_device_id,
                        "position": None,
                        "distance_from_camera": None,
                        "projected_position_2d_coordinates": None,
                        "distance_from_image_center": None,
                        "in_frame": None,
                        "in_middle": None,
                    }
                ]
            ).set_index(["camera_device_id"])

        return best_camera_view.sort_values("distance_from_image_center").iloc[:1]

    def best_camera_view_device_id(self):
        df_best_camera_view = self.best_camera_view()
        return df_best_camera_view.index[0]
