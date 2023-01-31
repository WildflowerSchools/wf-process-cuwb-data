import datetime

import cv_utils
import honeycomb_io
import numpy as np
import pandas as pd

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
        imputed_z_position=1.0,
        chunk_size=100,
        client=None,
        uri=None,
        token_uri=None,
        audience=None,
        client_id=None,
        client_secret=None,
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

        client_params = {
            "chunk_size": chunk_size,
            "client": client,
            "uri": uri,
            "token_uri": token_uri,
            "audience": audience,
            "client_id": client_id,
            "client_secret": client_secret,
        }

        if camera_device_ids is None:
            camera_info = honeycomb_caching_client.fetch_camera_devices(
                environment_id=environment_id, environment_name=environment_name, start=timestamp, end=timestamp
            )
            camera_device_ids = camera_info.index.unique().tolist()
        if camera_calibrations is None:
            camera_calibrations = honeycomb_caching_client.fetch_camera_calibrations(
                camera_ids=camera_device_ids, start=timestamp, end=timestamp
            )
        position_window_start = timestamp - datetime.timedelta(seconds=position_window_seconds / 2)
        position_window_end = timestamp + datetime.timedelta(seconds=position_window_seconds / 2)
        position_data = honeycomb_io.fetch_cuwb_position_data(
            start=position_window_start,
            end=position_window_end,
            device_ids=[tag_device_id],
            environment_id=None,
            environment_name=None,
            device_types=["UWBTAG"],
            output_format="dataframe",
            sort_arguments=None,
            **client_params,
        )
        if len(position_data) == 0:
            err = f"Unable to find position data between {position_window_start} and {position_window_end}, cannot determine best camera views"
            logger.warning(err)
            raise ValueError(err)

        position = np.nanmedian(position_data.loc[:, ["x", "y", "z"]].values, axis=0)
        if imputed_z_position is not None:
            position[2] = imputed_z_position

        view_data_list = []
        for camera_device_id, camera_calibration in camera_calibrations.items():
            camera_position = cv_utils.extract_camera_position(
                rotation_vector=camera_calibration["rotation_vector"],
                translation_vector=camera_calibration["translation_vector"],
            )
            distance_from_camera = np.linalg.norm(np.subtract(position, camera_position))
            image_position = cv_utils.project_points(
                object_points=position,
                rotation_vector=camera_calibration["rotation_vector"],
                translation_vector=camera_calibration["translation_vector"],
                camera_matrix=camera_calibration["camera_matrix"],
                distortion_coefficients=camera_calibration["distortion_coefficients"],
                remove_behind_camera=True,
                remove_outside_frame=True,
                image_corners=np.asarray(
                    [[0.0, 0.0], [camera_calibration["image_width"], camera_calibration["image_height"]]]
                ),
            )
            image_position = np.squeeze(image_position)
            if np.all(np.isfinite(image_position)):
                in_frame = True
                distance_from_image_center = np.linalg.norm(
                    np.subtract(
                        image_position, [camera_calibration["image_width"] / 2, camera_calibration["image_height"] / 2]
                    )
                )
                in_middle = (
                    image_position[0] > camera_calibration["image_width"] * (1.0 / 10.0)
                    and image_position[0] < camera_calibration["image_width"] * (9.0 / 10.0)
                    and image_position[1] > camera_calibration["image_height"] * (1.0 / 10.0)
                    and image_position[1] < camera_calibration["image_height"] * (9.0 / 10.0)
                )
            else:
                in_frame = False
                distance_from_image_center = None
                in_middle = False
            view_data_list.append(
                {
                    "camera_device_id": camera_device_id,
                    "position": position,
                    "distance_from_camera": distance_from_camera,
                    "image_position": image_position,
                    "distance_from_image_center": distance_from_image_center,
                    "in_frame": in_frame,
                    "in_middle": in_middle,
                }
            )
        df_view_data = pd.DataFrame(view_data_list).set_index("camera_device_id")
        df_view_data = df_view_data.sort_values("distance_from_camera")
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
        if not self.df_view_data["in_frame"].any():
            best_camera_view = pd.DataFrame(
                [
                    {
                        "camera_device_id": self.default_camera_device_id,
                        "position": None,
                        "distance_from_camera": None,
                        "image_position": None,
                        "distance_from_image_center": None,
                        "in_frame": None,
                        "in_middle": None,
                    }
                ]
            ).set_index(["camera_device_id"])
        elif not self.df_view_data["in_middle"].any():
            best_camera_view = self.df_view_data.sort_values("distance_from_image_center")
        else:
            best_camera_view = self.df_view_data.loc[self.df_view_data["in_middle"]]

        return best_camera_view.sort_values("distance_from_camera").iloc[:1]

    def best_camera_view_device_id(self):
        df_best_camera_view = self.best_camera_view()
        return df_best_camera_view.index[0]
